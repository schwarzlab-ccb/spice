#!/usr/bin/env python
"""Test suite for SPICE command-line interface."""

import os
import sys
import subprocess
import tempfile
import shutil
import pytest
import yaml

import pandas as pd


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with necessary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create necessary subdirectories
        data_dir = os.path.join(tmpdir, 'data')
        results_dir = os.path.join(tmpdir, 'results')
        logs_dir = os.path.join(tmpdir, 'logs')
        plot_dir = os.path.join(tmpdir, 'plots')
        os.makedirs(data_dir)
        os.makedirs(results_dir)
        os.makedirs(logs_dir)
        os.makedirs(plot_dir)
        
        # Copy example data file
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        example_data = os.path.join(repo_root, 'data', 'example_data.tsv')
        if os.path.exists(example_data):
            shutil.copy(example_data, os.path.join(data_dir, 'example_data.tsv'))

        example_sv_data = os.path.join(repo_root, 'data', 'example_sv_data.tsv')
        if os.path.exists(example_sv_data):
            shutil.copy(example_sv_data, os.path.join(data_dir, 'example_sv_data.tsv'))

        total_cn_data = os.path.join(repo_root, 'data', 'test_data_total_cn_small.tsv')
        total_cn_input = os.path.join(data_dir, 'test_data_total_cn_small.tsv')
        if os.path.exists(total_cn_data):
            shutil.copy(total_cn_data, total_cn_input)
        
        # Create a minimal config file
        config = {
            'name': 'test_run',
            'input_files': {
                'copynumber': os.path.join(data_dir, 'example_data.tsv'),
                'sv': os.path.join(data_dir, 'example_sv_data.tsv'),
            },
            'directories': {
                'base_dir': tmpdir,
                'results_dir': results_dir,
                'log_dir': logs_dir,
                'plot_dir': plot_dir,
            },
            'params': {
                'wgd_inference_method': 'ploidy_loh',
                'dist_limit': 40,
                'full_path_dist_limit': 9,
                'knn_k': 250,
                'sv_matching_threshold': 10,
                'time_limit_all_solutions': 60,
                'all_loh_solutions': False,
                'use_cache': True,
                'logging_level': 'INFO',
                'mcmc_n_iterations_scale': 1.0
            }
        }
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)

        total_cn_config = dict(config)
        total_cn_config['input_files'] = dict(config['input_files'])
        total_cn_config['params'] = dict(config['params'])
        total_cn_config['directories'] = dict(config['directories'])
        total_cn_config['input_files']['copynumber'] = total_cn_input
        total_cn_config['params']['total_cn'] = True

        total_cn_config_path = os.path.join(tmpdir, 'test_config_total_cn.yaml')
        with open(total_cn_config_path, 'w') as f:
            yaml.safe_dump(total_cn_config, f)
        
        yield tmpdir, config_path


class TestCLIBasic:
    """Basic CLI argument parsing tests (no execution)."""
    
    def test_spice_help_flag(self):
        """Test that spice --help works and shows main modes."""
        result = subprocess.run(
            ['spice', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'SPICE' in result.stdout
        assert 'event_inference' in result.stdout
        assert 'plotting' in result.stdout
        assert 'loci_detection' in result.stdout
        # Regression guard: the epilog's plotting examples must use the real,
        # registered flag names, not the stale --plot-sample/--plot-id.
        assert '--plot-sample' not in result.stdout
        assert '--plot-id' not in result.stdout
        assert '--plot-events-per-sample' in result.stdout
        assert '--plot-events-per-id' in result.stdout
    
    def test_spice_no_mode_fails(self):
        """Test that spice without mode argument fails."""
        result = subprocess.run(
            ['spice'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
    
    def test_spice_invalid_mode_fails(self):
        """Test that spice with invalid mode fails with clear error."""
        result = subprocess.run(
            ['spice', 'invalid_mode', '--config', 'dummy.yaml'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert 'invalid choice' in result.stderr.lower()
    
    def test_event_inference_help_flag(self):
        """Test that event_inference mode shows help."""
        result = subprocess.run(
            ['spice', 'event_inference', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'event' in result.stdout.lower()
        assert '--event-steps' in result.stdout
    
    def test_plotting_help_flag(self):
        """Test that plotting mode shows help."""
        result = subprocess.run(
            ['spice', 'plotting', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'plot' in result.stdout.lower()
    
    def test_loci_detection_help_flag(self):
        """Test that loci_detection mode shows help."""
        result = subprocess.run(
            ['spice', 'loci_detection', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'loci' in result.stdout.lower()
    
    def test_event_inference_config_required(self, temp_workspace):
        """Test that --config is required for event_inference."""
        result = subprocess.run(
            ['spice', 'event_inference'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert '--config' in result.stderr or 'required' in result.stderr.lower()
    
    def test_plotting_config_required(self):
        """Test that --config is required for plotting."""
        result = subprocess.run(
            ['spice', 'plotting', '--plot-events-per-sample', 'sample1'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
    
    def test_config_file_not_found(self):
        """Test that missing config file produces error."""
        result = subprocess.run(
            ['spice', 'event_inference', '--config', '/nonexistent/config.yaml'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
    
    def test_invalid_event_step_rejected(self, temp_workspace):
        """Test that invalid --event-steps names are rejected."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--event-steps', 'invalid_step', '--config', config_path],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert 'invalid' in result.stderr.lower()
    
    def test_valid_event_steps_accepted(self, temp_workspace):
        """Test that valid --event-steps names are accepted."""
        tmpdir, config_path = temp_workspace
        
        valid_steps = ['split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
        for step in valid_steps:
            result = subprocess.run(
                ['spice', 'event_inference', '--event-steps', step, '--config', config_path, '--clean'],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert 'Invalid step' not in result.stderr, f"Step '{step}' was rejected but should be valid"
    
    def test_event_steps_with_plus_suffix_accepted(self, temp_workspace):
        """Test that --event-steps with + suffix are accepted."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--event-steps', 'split+', '--config', config_path, '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert 'Invalid step' not in result.stderr
    
    def test_multiple_event_steps_accepted(self, temp_workspace):
        """Test that multiple --event-steps can be specified."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--event-steps', 'split', 'disambiguate', '--config', config_path, '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert 'Invalid step' not in result.stderr
    
    def test_event_inference_cores_flag(self, temp_workspace):
        """Test that --cores flag is accepted for event_inference."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--cores', '4', '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0 or 'cores' not in result.stderr.lower()
    
    def test_log_flag_options(self, temp_workspace):
        """Test that --log flag accepts valid options."""
        tmpdir, config_path = temp_workspace
        
        for log_mode in ['terminal', 'file', 'both']:
            result = subprocess.run(
                ['spice', 'event_inference', '--config', config_path, '--log', log_mode, '--clean'],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0 or 'log' not in result.stderr.lower()
    
    def test_debug_flag_accepted(self, temp_workspace):
        """Test that --debug flag is accepted."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--debug', '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0 or 'debug' not in result.stderr.lower()
    
    def test_keep_old_flag_accepted(self, temp_workspace):
        """Test that --keep-old flag is accepted for event_inference."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--keep-old', '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0 or 'keep-old' not in result.stderr.lower()
    
    def test_ids_flag_accepted(self, temp_workspace):
        """Test that --ids flag is accepted for event_inference."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--ids', 'sample1,sample2', '--clean'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0 or 'ids' not in result.stderr.lower()
    
    def test_plotting_requires_plot_target(self):
        """Test that plotting mode requires --plot-sample or --plot-id."""
        result = subprocess.run(
            ['spice', 'plotting', '--config', 'dummy.yaml'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
    
    def test_plotting_plot_sample_accepted(self, temp_workspace):
        """Test that --plot-events-per-sample is accepted for plotting mode."""
        tmpdir, config_path = temp_workspace

        result = subprocess.run(
            ['spice', 'plotting', '--config', config_path, '--plot-events-per-sample', 'sample1'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Argument parsing should succeed even if plotting execution fails later
        # (e.g. because event_inference hasn't been run yet in this workspace).
        assert 'unrecognized arguments' not in result.stderr
        assert 'one of the arguments' not in result.stderr

    def test_plotting_plot_id_accepted(self, temp_workspace):
        """Test that --plot-events-per-id is accepted for plotting mode."""
        tmpdir, config_path = temp_workspace

        result = subprocess.run(
            ['spice', 'plotting', '--config', config_path, '--plot-events-per-id', 'sample:chr1:cn_a'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Argument parsing should succeed even if plotting execution fails later
        # (e.g. because event_inference hasn't been run yet in this workspace).
        assert 'unrecognized arguments' not in result.stderr
        assert 'one of the arguments' not in result.stderr


class TestEventInferenceExecution:
    """Event inference mode execution tests."""

    def test_normal_execution(self, temp_workspace):
        """Test basic event_inference execution."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Verify expected outputs exist
        results_dir = os.path.join(tmpdir, 'results', 'test_run')
        assert os.path.exists(os.path.join(results_dir, 'final_events.tsv')), "final_events.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events_summary.tsv')), "events_summary.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events', 'failed_reports.tsv')), "failed_reports.tsv was not created"
        # Ensure that no ID failed
        with open(os.path.join(results_dir, 'events', 'failed_reports.tsv'), 'r') as f:
            content = f.readlines()
            assert len(content) == 1, f"Expected only header in failed_reports.tsv but found {len(content)-1} entries"
        
        # Check that SV-based solution was found
        final_events_df = pd.read_csv(os.path.join(results_dir, 'final_events.tsv'), sep='\t')
        assert final_events_df.query('id == "sample_13:chr12:cn_b"')['solved'].unique()[0] == 'sv'  
  
        # Ensure the debug sample ID appears in the failure report
        # with open(os.path.join(results_dir, 'events', 'failed_reports.tsv'), 'r') as f:
        #     content = f.read()
        #     assert 'sample_DEBUG' in content

    def test_normal_execution_total_cn(self, temp_workspace):
        """Test basic event_inference execution with total_cn mode enabled."""
        pytest.importorskip('fstlib')
        tmpdir, config_path = temp_workspace

        total_cn_input = os.path.join(tmpdir, 'data', 'test_data_total_cn_small.tsv')
        assert os.path.exists(total_cn_input), f"Missing test file: {total_cn_input}"
        total_cn_config_path = os.path.join(tmpdir, 'test_config_total_cn.yaml')
        assert os.path.exists(total_cn_config_path), f"Missing total_cn config: {total_cn_config_path}"

        result = subprocess.run(
            ['spice', 'event_inference', '--config', total_cn_config_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"total_cn event inference failed: {result.stderr}"

        results_dir = os.path.join(tmpdir, 'results', 'test_run')
        assert os.path.exists(os.path.join(results_dir, 'final_events.tsv')), "final_events.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events_summary.tsv')), "events_summary.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events', 'failed_reports.tsv')), "failed_reports.tsv was not created"
        # Ensure that no ID failed
        with open(os.path.join(results_dir, 'events', 'failed_reports.tsv'), 'r') as f:
            content = f.readlines()
            assert len(content) == 1, f"Expected only header in failed_reports.tsv but found {len(content)-1} entries"

    def test_execution_with_cores(self, temp_workspace):
        """Test event_inference execution with multiple cores."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--cores', '4'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Verify expected outputs exist
        results_dir = os.path.join(tmpdir, 'results', 'test_run')
        assert os.path.exists(os.path.join(results_dir, 'final_events.tsv')), "final_events.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events_summary.tsv')), "events_summary.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events', 'failed_reports.tsv')), "failed_reports.tsv was not created"
        # Ensure that no ID failed
        with open(os.path.join(results_dir, 'events', 'failed_reports.tsv'), 'r') as f:
            content = f.readlines()
            assert len(content) == 1, f"Expected only header in failed_reports.tsv but found {len(content)-1} entries"
        
        # Check that SV-based solution was found
        final_events_df = pd.read_csv(os.path.join(results_dir, 'final_events.tsv'), sep='\t')
        assert final_events_df.query('id == "sample_13:chr12:cn_b"')['solved'].unique()[0] == 'sv'  


    def test_execution_with_extra_preprocessing(self, temp_workspace):
        """Test event_inference execution with extra preprocessing steps."""
        tmpdir, config_path = temp_workspace
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--run-preprocessing'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Verify expected outputs exist
        results_dir = os.path.join(tmpdir, 'results', 'test_run')
        assert os.path.exists(os.path.join(results_dir, 'final_events.tsv')), "final_events.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events_summary.tsv')), "events_summary.tsv was not created"
        assert os.path.exists(os.path.join(results_dir, 'events', 'failed_reports.tsv')), "failed_reports.tsv was not created"
        
        # Check that SV-based solution was found
        final_events_df = pd.read_csv(os.path.join(results_dir, 'final_events.tsv'), sep='\t')
        assert final_events_df.query('id == "sample_13:chr12:cn_b"')['solved'].unique()[0] == 'sv'  

        # Ensure that no ID failed
        with open(os.path.join(results_dir, 'events', 'failed_reports.tsv'), 'r') as f:
            content = f.readlines()
            assert len(content) == 1, f"Expected only header in failed_reports.tsv but found {len(content)-1} entries"

    def test_clean_flag(self, temp_workspace):
        """Test that --clean flag works and cleans directories."""
        tmpdir, config_path = temp_workspace
        results_dir = os.path.join(tmpdir, 'results', 'test_run', 'events')
        
        # Create some dummy directories
        os.makedirs(os.path.join(results_dir, 'wgd'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'nowgd'), exist_ok=True)
        dummy_file = os.path.join(results_dir, 'wgd', 'test.txt')
        with open(dummy_file, 'w') as f:
            f.write('test')
        
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path, '--clean'],
            capture_output=True,
            text=True
        )
        
        # Should execute successfully
        assert result.returncode == 0
        # Directories should be cleaned
        assert not os.path.exists(dummy_file)


class TestPlottingExecution:
    """Plotting mode execution tests."""

    def test_plotting_with_plot_sample(self, temp_workspace):
        """Test plotting mode with --plot-sample."""
        tmpdir, config_path = temp_workspace
        
        # First run event_inference to generate required files
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Event inference failed: {result.stderr}"
        
        # Now test plotting
        result = subprocess.run(
            ['spice', 'plotting', '--config', config_path, '--plot-events-per-sample', 'sample_1'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Plotting execution failed: {result.stderr}"
        plots_dir = os.path.join(tmpdir, 'plots', 'test_run')
        plot_file = os.path.join(plots_dir, 'sample_1_events.png')
        assert os.path.exists(plot_file), f"Plot was not created ({plot_file})"

    def test_plotting_with_plot_id(self, temp_workspace):
        """Test plotting mode with --plot-id."""
        tmpdir, config_path = temp_workspace
        
        # First run event_inference to generate required files
        result = subprocess.run(
            ['spice', 'event_inference', '--config', config_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Event inference failed: {result.stderr}"
        
        # Now test plotting with plot-id
        result = subprocess.run(
            ['spice', 'plotting', '--config', config_path, '--plot-events-per-id', 'sample_1:chr1:cn_a'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Plotting execution failed: {result.stderr}"
        plots_dir = os.path.join(tmpdir, 'plots', 'test_run')
        plot_file = os.path.join(plots_dir, 'sample_1_chr1_cn_a_events.png')
        assert os.path.exists(plot_file), f"Plot was not created ({plot_file})"

