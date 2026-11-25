import re

import pytest
from requests.exceptions import HTTPError

from mcp_jenkins.jenkins import PendingInputsUnavailableError
from mcp_jenkins.jenkins.model.build import Artifact, Build, BuildReplay, PendingInput
from mcp_jenkins.server import build


@pytest.fixture
def mock_jenkins(mocker):
    mock_jenkins = mocker.Mock()

    mocker.patch('mcp_jenkins.server.build.jenkins', return_value=mock_jenkins)

    yield mock_jenkins


@pytest.mark.asyncio
async def test_get_running_builds(mock_jenkins, mocker):
    build1 = Build(number=1, url='1', building=True, timestamp=1234567890)
    build2 = Build(number=2, url='2', building=True, timestamp=1234567891)
    mock_jenkins.get_running_builds.return_value = [build1, build2]

    assert await build.get_running_builds(mocker.Mock()) == [
        {'number': 1, 'url': '1', 'building': True, 'timestamp': 1234567890},
        {'number': 2, 'url': '2', 'building': True, 'timestamp': 1234567891},
    ]


@pytest.mark.asyncio
async def test_get_build(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build.return_value = Build(number=1, url='1', building=False, timestamp=1234567890)

    assert await build.get_build(mocker.Mock(), fullname='job1') == {
        'number': 1,
        'url': '1',
        'building': False,
        'timestamp': 1234567890,
    }


@pytest.mark.asyncio
async def test_get_build_scripts(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_replay.return_value = BuildReplay(scripts=['script1', 'script2'])

    assert await build.get_build_scripts(mocker.Mock(), fullname='job1') == [
        'script1',
        'script2',
    ]


@pytest.mark.asyncio
async def test_get_build_console_output(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_console_output.return_value = 'Console output here'

    assert await build.get_build_console_output(mocker.Mock(), fullname='job1') == 'Console output here'
    mock_jenkins.get_build_console_output.assert_called_once_with(
        fullname='job1', number=1, pattern=None, offset=0, limit=None
    )


@pytest.mark.asyncio
async def test_get_build_console_output_with_number(mock_jenkins, mocker):
    mock_jenkins.get_build_console_output.return_value = 'output'

    assert await build.get_build_console_output(mocker.Mock(), fullname='job1', number=5) == 'output'
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_console_output.assert_called_once_with(
        fullname='job1', number=5, pattern=None, offset=0, limit=None
    )


@pytest.mark.asyncio
async def test_get_build_console_output_with_all_params(mock_jenkins, mocker):
    mock_jenkins.get_build_console_output.return_value = 'ERROR: boom'

    result = await build.get_build_console_output(
        mocker.Mock(), fullname='job1', number=3, pattern='ERROR', offset=1, limit=10
    )
    assert result == 'ERROR: boom'
    mock_jenkins.get_build_console_output.assert_called_once_with(
        fullname='job1', number=3, pattern='ERROR', offset=1, limit=10
    )


@pytest.mark.asyncio
async def test_get_build_console_output_no_build(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = None

    with pytest.raises(ValueError, match='No build found for job: job1'):
        await build.get_build_console_output(mocker.Mock(), fullname='job1')


def _write_download(content: bytes):
    def download(*, path, **_kwargs: object):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return len(content)

    return download


@pytest.mark.asyncio
async def test_download_build_console_output(mock_jenkins, mocker, tmp_path):
    content = b'first\r\nERROR: a\nmid\rstill mid\nERROR: b\nlast\n'
    mock_jenkins.download_build_console_output.side_effect = _write_download(content)
    path = tmp_path / 'console.log'

    result = await build.download_build_console_output(
        mocker.Mock(), fullname='job1', number=3, path=str(path), pattern='ERROR', tail=2
    )

    assert result == {
        'number': 3,
        'path': str(path.resolve()),
        'size': len(content),
        'lineCount': 5,
        'tail': ['ERROR: b', 'last'],
        'matchCount': 2,
        'matches': [{'line': 2, 'text': 'ERROR: a'}, {'line': 4, 'text': 'ERROR: b'}],
    }
    mock_jenkins.download_build_console_output.assert_called_once_with(fullname='job1', number=3, path=path.resolve())


@pytest.mark.asyncio
async def test_download_build_console_output_default_path(mock_jenkins, mocker, tmp_path):
    mocker.patch.object(build, 'DOWNLOAD_DIR', tmp_path)
    mock_jenkins.get_last_build_number.return_value = 7
    mock_jenkins.download_build_console_output.side_effect = _write_download(b'only line')

    result = await build.download_build_console_output(mocker.Mock(), fullname='folder/job1')

    expected_path = tmp_path / 'folder' / 'job1' / '7' / 'console.log'
    assert result == {'number': 7, 'path': str(expected_path), 'size': 9, 'lineCount': 1, 'tail': ['only line']}


@pytest.mark.asyncio
async def test_download_build_console_output_limits_matches_and_cuts_long_lines(mock_jenkins, mocker, tmp_path):
    long_line = 'E' * (build.SUMMARY_LINE_LENGTH + 5)
    content = '\n'.join(['E1', 'E2', 'E3', long_line]).encode()
    mock_jenkins.download_build_console_output.side_effect = _write_download(content)

    result = await build.download_build_console_output(
        mocker.Mock(),
        fullname='job1',
        number=1,
        path=str(tmp_path / 'console.log'),
        pattern='E',
        tail=1,
        max_matches=2,
    )

    assert result['matchCount'] == 4
    assert result['matches'] == [{'line': 1, 'text': 'E1'}, {'line': 2, 'text': 'E2'}]
    assert result['tail'] == ['E' * build.SUMMARY_LINE_LENGTH + '... [5 more characters]']


@pytest.mark.asyncio
async def test_download_build_console_output_invalid_pattern(mock_jenkins, mocker, tmp_path):
    with pytest.raises(re.error):
        await build.download_build_console_output(
            mocker.Mock(), fullname='job1', number=1, path=str(tmp_path / 'console.log'), pattern='('
        )

    mock_jenkins.download_build_console_output.assert_not_called()


@pytest.mark.asyncio
async def test_get_build_test_reports(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_test_report.return_value = {'reports': ['report1', 'report2']}

    assert await build.get_build_test_report(mocker.Mock(), fullname='job1') == {'reports': ['report1', 'report2']}


@pytest.mark.asyncio
async def test_get_build_test_failures(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 7
    mock_jenkins.get_build_test_failures.return_value = {'failCount': 0, 'truncated': False, 'failures': []}

    assert await build.get_build_test_failures(mocker.Mock(), fullname='job1', limit=10) == {
        'failCount': 0,
        'truncated': False,
        'failures': [],
    }
    mock_jenkins.get_build_test_failures.assert_called_once_with(
        fullname='job1',
        number=7,
        limit=10,
        max_stack_trace_length=2000,
    )


@pytest.mark.asyncio
async def test_get_build_parameters(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_parameters.return_value = {'BRANCH': 'main', 'DEBUG': True}

    assert await build.get_build_parameters(mocker.Mock(), fullname='job1') == {
        'BRANCH': 'main',
        'DEBUG': True,
    }


@pytest.mark.asyncio
async def test_stop_build(mock_jenkins, mocker):
    await build.stop_build(mocker.Mock(), fullname='job1', number=1)
    mock_jenkins.stop_build.assert_called_once_with(fullname='job1', number=1)


@pytest.mark.asyncio
async def test_get_pending_inputs(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(
            id='Deploy',
            message='Deploy to prod?',
            proceedText='Deploy',
            inputs=[{'name': 'TARGET', 'type': 'StringParameterDefinition'}],
        )
    ]

    assert await build.get_pending_inputs(mocker.Mock(), fullname='job1') == [
        {
            'id': 'Deploy',
            'message': 'Deploy to prod?',
            'proceedText': 'Deploy',
            'inputs': [{'name': 'TARGET', 'type': 'StringParameterDefinition'}],
        }
    ]
    mock_jenkins.get_build_pending_inputs.assert_called_once_with(fullname='job1', number=1)


@pytest.mark.asyncio
async def test_get_pending_inputs_with_number(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy')]

    assert await build.get_pending_inputs(mocker.Mock(), fullname='job1', number=5) == [{'id': 'Deploy', 'inputs': []}]
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_pending_inputs.assert_called_once_with(fullname='job1', number=5)


@pytest.mark.asyncio
async def test_get_pending_inputs_empty(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = []

    assert await build.get_pending_inputs(mocker.Mock(), fullname='job1', number=5) == []


@pytest.mark.asyncio
async def test_get_pending_inputs_no_build(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = None

    with pytest.raises(ValueError, match='No build found for job: job1'):
        await build.get_pending_inputs(mocker.Mock(), fullname='job1')

    mock_jenkins.get_build_pending_inputs.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_auto_resolves_input_id(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy')]

    assert await build.submit_input(mocker.Mock(), fullname='job1', number=1) == {
        'fullname': 'job1',
        'number': 1,
        'inputId': 'Deploy',
        'action': 'proceedEmpty',
    }
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=1, input_id='Deploy', action='proceedEmpty', parameters=None
    )


@pytest.mark.asyncio
async def test_submit_input_with_parameters(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(
            id='Deploy',
            inputs=[
                {'name': 'APPROVE', 'type': 'BooleanParameterDefinition'},
                {'name': 'TARGET', 'type': 'StringParameterDefinition'},
            ],
        )
    ]

    await build.submit_input(mocker.Mock(), fullname='job1', number=1, parameters={'APPROVE': True, 'TARGET': 'prod'})

    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1',
        number=1,
        input_id='Deploy',
        action='proceed',
        parameters={'APPROVE': True, 'TARGET': 'prod'},
    )


@pytest.mark.asyncio
async def test_submit_input_rejects_undeclared_parameter(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'APPROVED', 'type': 'BooleanParameterDefinition'}])
    ]

    with pytest.raises(ValueError, match='does not declare APPROVE') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=1, parameters={'APPROVE': True})

    assert 'Declared parameters: APPROVED' in str(exc_info.value)
    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_rejects_parameters_on_parameterless_input(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy')]

    with pytest.raises(ValueError, match='does not declare APPROVE') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=1, parameters={'APPROVE': True})

    assert 'Declared parameters: none' in str(exc_info.value)
    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_rejects_partial_parameters(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(
            id='Deploy',
            inputs=[
                {'name': 'CONFIRM', 'type': 'BooleanParameterDefinition'},
                {'name': 'TARGET', 'type': 'StringParameterDefinition'},
            ],
        )
    ]

    with pytest.raises(ValueError, match='no value for CONFIRM'):
        await build.submit_input(mocker.Mock(), fullname='job1', number=1, parameters={'TARGET': 'prod'})

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_refuses_to_proceed_empty_on_declared_parameters(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'ENV', 'type': 'ChoiceParameterDefinition'}])
    ]

    with pytest.raises(ValueError, match='declares ENV'):
        await build.submit_input(mocker.Mock(), fullname='job1', number=1)

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_aborts_declared_parameter_input_without_values(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'ENV', 'type': 'ChoiceParameterDefinition'}])
    ]

    await build.submit_input(mocker.Mock(), fullname='job1', number=1, action='abort')

    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=1, input_id='Deploy', action='abort', parameters=None
    )


@pytest.mark.asyncio
async def test_submit_input_accepts_declared_parameters(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'APPROVED', 'type': 'BooleanParameterDefinition'}])
    ]

    result = await build.submit_input(mocker.Mock(), fullname='job1', number=1, parameters={'APPROVED': True})

    assert result['action'] == 'proceed'
    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=1, input_id='Deploy', action='proceed', parameters={'APPROVED': True}
    )


@pytest.mark.asyncio
async def test_submit_input_validates_names_with_explicit_input_id(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'TARGET', 'type': 'StringParameterDefinition'}])
    ]

    with pytest.raises(ValueError, match='does not declare ANYTHING'):
        await build.submit_input(
            mocker.Mock(), fullname='job1', number=7, input_id='Deploy', parameters={'ANYTHING': True}
        )

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_explicit_input_id_refuses_to_proceed_empty_on_declared_parameters(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [
        PendingInput(id='Deploy', inputs=[{'name': 'TARGET', 'type': 'StringParameterDefinition'}])
    ]

    with pytest.raises(ValueError, match='no value for TARGET'):
        await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy')

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_explicit_input_id_not_pending(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Rollback')]

    with pytest.raises(ValueError, match='No pending input Deploy') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy')

    assert 'Rollback' in str(exc_info.value)
    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_explicit_input_id_without_wfapi_submits_unvalidated(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.side_effect = PendingInputsUnavailableError(
        'No wfapi pending input endpoint for job1 #7'
    )

    await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy', parameters={'ANYTHING': True})

    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=7, input_id='Deploy', action='proceed', parameters={'ANYTHING': True}
    )


@pytest.mark.asyncio
async def test_submit_input_explicit_input_id_wfapi_http_error_submits_unvalidated(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.side_effect = HTTPError('500 Server Error')

    await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy', parameters={'ANYTHING': True})

    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=7, input_id='Deploy', action='proceed', parameters={'ANYTHING': True}
    )


@pytest.mark.asyncio
async def test_submit_input_malformed_wfapi_response_does_not_skip_validation(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.side_effect = ValueError('Expecting value: line 1 column 1 (char 0)')

    with pytest.raises(ValueError, match='Expecting value'):
        await build.submit_input(
            mocker.Mock(), fullname='job1', number=7, input_id='Deploy', parameters={'ANYTHING': True}
        )

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_failed_submit_reports_discovery_error(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.side_effect = PendingInputsUnavailableError(
        'No wfapi pending input endpoint for job1 #999'
    )
    mock_jenkins.submit_build_input.side_effect = HTTPError('404 Client Error')

    with pytest.raises(ValueError, match='No wfapi pending input endpoint') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=999, input_id='Deploy')

    assert '404 Client Error' in str(exc_info.value)


@pytest.mark.asyncio
async def test_submit_input_failed_submit_hints_at_settled_input(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy')]
    mock_jenkins.submit_build_input.side_effect = HTTPError('400 Client Error')

    with pytest.raises(ValueError, match='may already be settled') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=1)

    assert '400 Client Error' in str(exc_info.value)


@pytest.mark.asyncio
async def test_submit_input_with_explicit_input_id(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy')]

    await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy')

    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_pending_inputs.assert_called_once_with(fullname='job1', number=7)
    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=7, input_id='Deploy', action='proceedEmpty', parameters=None
    )


@pytest.mark.asyncio
async def test_submit_input_abort(mock_jenkins, mocker):
    assert await build.submit_input(mocker.Mock(), fullname='job1', number=7, input_id='Deploy', action='abort') == {
        'fullname': 'job1',
        'number': 7,
        'inputId': 'Deploy',
        'action': 'abort',
    }
    mock_jenkins.get_build_pending_inputs.assert_not_called()
    mock_jenkins.submit_build_input.assert_called_once_with(
        fullname='job1', number=7, input_id='Deploy', action='abort', parameters=None
    )


@pytest.mark.asyncio
async def test_submit_input_abort_with_parameters(mock_jenkins, mocker):
    with pytest.raises(ValueError, match='cannot be combined'):
        await build.submit_input(
            mocker.Mock(), fullname='job1', number=7, input_id='Deploy', action='abort', parameters={'A': 'b'}
        )

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_no_pending_inputs(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = []

    with pytest.raises(ValueError, match='No pending input for job1 #1'):
        await build.submit_input(mocker.Mock(), fullname='job1', number=1)

    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_submit_input_multiple_pending_inputs(mock_jenkins, mocker):
    mock_jenkins.get_build_pending_inputs.return_value = [PendingInput(id='Deploy'), PendingInput(id='Rollback')]

    with pytest.raises(ValueError, match='Multiple pending inputs') as exc_info:
        await build.submit_input(mocker.Mock(), fullname='job1', number=1)

    assert 'Deploy' in str(exc_info.value)
    assert 'Rollback' in str(exc_info.value)
    mock_jenkins.submit_build_input.assert_not_called()


@pytest.mark.asyncio
async def test_get_all_build_artifacts(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_artifacts.return_value = [
        Artifact(
            fileName='index.html',
            relativePath='playwright-report/index.html',
            displayPath='playwright-report/index.html',
        ),
        Artifact(fileName='trace.zip', relativePath='trace.zip', displayPath='trace.zip'),
    ]

    assert await build.get_all_build_artifacts(mocker.Mock(), fullname='job1') == [
        {
            'fileName': 'index.html',
            'relativePath': 'playwright-report/index.html',
            'displayPath': 'playwright-report/index.html',
        },
        {'fileName': 'trace.zip', 'relativePath': 'trace.zip', 'displayPath': 'trace.zip'},
    ]


@pytest.mark.asyncio
async def test_get_all_build_artifacts_with_number(mock_jenkins, mocker):
    mock_jenkins.get_build_artifacts.return_value = []

    assert await build.get_all_build_artifacts(mocker.Mock(), fullname='job1', number=5) == []
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_artifacts.assert_called_once_with(fullname='job1', number=5)


@pytest.mark.asyncio
async def test_get_build_artifact_text(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_artifact.return_value = b'<html>report</html>'

    result = await build.get_build_artifact(
        mocker.Mock(), fullname='job1', relative_path='playwright-report/index.html'
    )
    assert result == {'content': '<html>report</html>', 'encoding': 'utf-8'}


@pytest.mark.asyncio
async def test_get_build_artifact_binary(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_artifact.return_value = bytes(range(256))

    result = await build.get_build_artifact(mocker.Mock(), fullname='job1', relative_path='trace.zip')
    assert result['encoding'] == 'base64'
    import base64

    assert base64.b64decode(result['content']) == bytes(range(256))


@pytest.mark.asyncio
async def test_get_build_artifact_with_number(mock_jenkins, mocker):
    mock_jenkins.get_build_artifact.return_value = b'data'

    result = await build.get_build_artifact(mocker.Mock(), fullname='job1', relative_path='file.txt', number=3)
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_artifact.assert_called_once_with(fullname='job1', number=3, relative_path='file.txt')
    assert result == {'content': 'data', 'encoding': 'utf-8'}


@pytest.mark.asyncio
async def test_get_build_artifact_url(mock_jenkins, mocker):
    mock_jenkins.get_last_build_number.return_value = 1
    mock_jenkins.get_build_artifact_url.return_value = 'https://jenkins.example.com/job/job1/1/artifact/trace.zip'

    result = await build.get_build_artifact_url(mocker.Mock(), fullname='job1', relative_path='trace.zip')
    assert result == 'https://jenkins.example.com/job/job1/1/artifact/trace.zip'


@pytest.mark.asyncio
async def test_get_build_artifact_url_with_number(mock_jenkins, mocker):
    mock_jenkins.get_build_artifact_url.return_value = 'https://jenkins.example.com/job/job1/5/artifact/report.html'

    result = await build.get_build_artifact_url(mocker.Mock(), fullname='job1', relative_path='report.html', number=5)
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.get_build_artifact_url.assert_called_once_with(fullname='job1', number=5, relative_path='report.html')
    assert result == 'https://jenkins.example.com/job/job1/5/artifact/report.html'


@pytest.mark.asyncio
async def test_download_build_artifact(mock_jenkins, mocker, tmp_path):
    mock_jenkins.download_build_artifact.side_effect = _write_download(b'zipdata')
    path = tmp_path / 'trace.zip'

    result = await build.download_build_artifact(
        mocker.Mock(), fullname='job1', relative_path='out/trace.zip', number=5, path=str(path)
    )

    assert result == {'number': 5, 'path': str(path.resolve()), 'size': 7}
    mock_jenkins.get_last_build_number.assert_not_called()
    mock_jenkins.download_build_artifact.assert_called_once_with(
        fullname='job1', number=5, relative_path='out/trace.zip', path=path.resolve()
    )


@pytest.mark.asyncio
async def test_download_build_artifact_default_path(mock_jenkins, mocker, tmp_path):
    mocker.patch.object(build, 'DOWNLOAD_DIR', tmp_path)
    mock_jenkins.get_last_build_number.return_value = 2
    mock_jenkins.download_build_artifact.side_effect = _write_download(b'zipdata')

    result = await build.download_build_artifact(mocker.Mock(), fullname='job1', relative_path='out/trace.zip')

    expected_path = tmp_path / 'job1' / '2' / 'artifacts' / 'out' / 'trace.zip'
    assert result == {'number': 2, 'path': str(expected_path), 'size': 7}
    assert expected_path.read_bytes() == b'zipdata'
