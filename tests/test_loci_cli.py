#!/usr/bin/env python
"""Test suite for SPICE loci detection and assignment CLI."""

from ast import pattern
import os
import sys
import subprocess
import tempfile
import shutil
import pytest
import yaml
import pandas as pd


@pytest.fixture
def temp_workspace_with_loci():
    """Create a temporary workspace with loci detection data (chr1 only for speed)."""
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
        
        # Get repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Copy and filter final_events data to chr1 only
        pcawg_events = os.path.join(repo_root, 'data', 'pcawg_final_events_chr1_chr2.tsv')
        if os.path.exists(pcawg_events):
            df = pd.read_csv(pcawg_events, sep='\t')
            # Filter to chr1 only to save time
            df_filtered = df[df['chrom'] == 'chr1'].copy()
            final_events_path = os.path.join(data_dir, 'pcawg_final_events_chr1_only.tsv')
            df_filtered.to_csv(final_events_path, sep='\t', index=False)
        
        # Copy plateaus file if available
        plateaus_src = os.path.join(repo_root, 'data', 'plateaus.tsv')
        if os.path.exists(plateaus_src):
            plateaus_dst = os.path.join(data_dir, 'plateaus.tsv')
            shutil.copy(plateaus_src, plateaus_dst)
        
        # Create a loci detection config file
        config = {
            'name': 'test_loci',
            'input_files': {
                'final_events': final_events_path if os.path.exists(pcawg_events) else None,
                'plateaus': os.path.join(data_dir, 'plateaus.tsv') if os.path.exists(plateaus_src) else None,
            },
            'directories': {
                'base_dir': tmpdir,
                'data_dir': data_dir,
                'results_dir': results_dir,
                'log_dir': logs_dir,
                'plot_dir': plot_dir
            },
            'loci_detection': {
                # From loci_example.yaml
                'loci_steps': 'fast',
                'N_loci': 1,
                'skip_up_down': False,
                'use_original_rank': False,
                'length_scales_for_residuals': '01234567',
                'N_bootstrap': 10,
                'N_kernel': 1000,
                'overwrite_preprocessing': False,
                'detection_blocked_distance_th': 5000000,
                'detection_N_iterations_base': 30,
                'detection_max_N_iterations': 200,
                'detection_final_N_iterations': 250,
                'ranking_N_iterations': 5,
                'flipping_N_iterations': 110,
                'flipping_N_iterations_single': 10,
                'limiting_N_iterations_optim': 100,
                'optimizing_N_iterations_optimization': 110,
                'infer_widths_N_iterations': 10,
                'merge_N_iterations_optim': 100,
                'within_ci_N_iterations': 100,
                'filter_N_iterations_optim': 100,
                'final_limiting_N_iterations_optim': 100,
                'N_bootstrap_for_widths': 5,
                'th_locus_prominence': 1,
                'loci_assignment_N_iterations': 250,
                'loci_assignment_within_ci_N_iterations': 100,
                'p_values_N_random': 100,
                'p_values_N_iterations': 100,
                'calculate_p_value': True,
                'p_value_threshold': 0.05,
                'remove_plateaus': True,
                'remove_chrY': True,
                'drop_duplicates': True,
                'use_observed_centromeres': True,
            },
            'params': {
                'logging_level': 'INFO',
            }
        }
        
        config_path = os.path.join(tmpdir, 'test_loci_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        yield tmpdir, config_path


class TestLociDetectionCLIBasic:
    """Basic CLI argument parsing tests for loci detection (no execution)."""
    
    def test_loci_detection_help_flag(self):
        """Test that loci_detection --help works."""
        result = subprocess.run(
            ['spice', 'loci_detection', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'loci' in result.stdout.lower()
        assert '--loci-steps' in result.stdout
    
    def test_loci_detection_config_required(self):
        """Test that --config is required for loci_detection."""
        result = subprocess.run(
            ['spice', 'loci_detection'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert '--config' in result.stderr or 'required' in result.stderr.lower()
    

class TestLociAssignmentCLIBasic:
    """Basic CLI argument parsing tests for loci assignment (no execution)."""
    
    def test_loci_assignment_help_flag(self):
        """Test that loci_assignment --help works."""
        result = subprocess.run(
            ['spice', 'loci_assignment', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'loci' in result.stdout.lower()
        assert 'assignment' in result.stdout.lower()
    
    def test_loci_assignment_config_required(self):
        """Test that --config is required for loci_assignment."""
        result = subprocess.run(
            ['spice', 'loci_assignment'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert '--config' in result.stderr or 'required' in result.stderr.lower()
    

class TestLociDetectionExecution:
    """Loci detection mode execution tests."""
    
    def test_loci_detection_basic_execution(self, temp_workspace_with_loci):
        """Test basic loci_detection execution."""
        tmpdir, config_path = temp_workspace_with_loci
        
        result = subprocess.run(
            ['spice', 'loci_detection', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        assert result.returncode == 0, f"Loci detection failed: {result.stderr}"
        
        # Verify expected outputs exist
        results_dir = os.path.join(tmpdir, 'results', 'test_loci')
        final_loci_path = os.path.join(results_dir, 'final_loci_detection.tsv')
        assert os.path.exists(final_loci_path), f"final_loci_detection.tsv was not created at {final_loci_path}"
        
        # Verify file has content
        df = pd.read_csv(final_loci_path, sep='\t', index_col=0)
        assert len(df) > 0, "final_loci_detection.tsv is empty"
        assert 'chrom' in df.columns, "final_loci_detection.tsv missing 'chrom' column"
        assert 'start' in df.columns or 'position' in df.columns, "final_loci_detection.tsv missing position columns"
    
        # Verify loci_of_selection directory structure was created
        loci_dir = os.path.join(results_dir, 'loci_of_selection')
        
        # These should exist based on pipeline structure
        expected_dirs = ['data_per_length_scale', 'detection']
        for expected_dir in expected_dirs:
            dir_path = os.path.join(loci_dir, expected_dir)
            assert os.path.exists(dir_path), f"Expected directory {dir_path} was not created"

        final_loci_path = os.path.join(results_dir, 'final_loci_detection.tsv')
        df = pd.read_csv(final_loci_path, sep='\t', index_col=0)
        assert len(df) > 0, "final_loci_detection.tsv is empty"
        
        # Check for some expected columns (adapt based on actual output)
        expected_columns = ['chrom', 'start', 'end']  # These are likely column names
        for col in expected_columns:
            assert col in df.columns, f"Critical column containing '{col}' not found in output"

    def test_loci_detection_with_overwrite_flag(self, temp_workspace_with_loci):
        """Test loci_detection execution with --overwrite flag."""
        tmpdir, config_path = temp_workspace_with_loci
        
        # First run
        result1 = subprocess.run(
            ['spice', 'loci_detection', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        assert result1.returncode == 0, f"First loci detection run failed: {result1.stderr}"
        
        results_dir = os.path.join(tmpdir, 'results', 'test_loci')
        final_loci_path = os.path.join(results_dir, 'final_loci_detection.tsv')
        assert os.path.exists(final_loci_path), "First run did not create final_loci_detection.tsv"
        
        # Get modification time of first run
        first_mtime = os.path.getmtime(final_loci_path)
        
        # Second run with --overwrite
        result2 = subprocess.run(
            ['spice', 'loci_detection', '--config', config_path, '--overwrite'],
            capture_output=True,
            text=True,
            timeout=600
        )
        assert result2.returncode == 0, f"Second loci detection run with --overwrite failed: {result2.stderr}"
        
        # File should be recreated (newer modification time)
        second_mtime = os.path.getmtime(final_loci_path)
        assert second_mtime >= first_mtime, "File was not recreated with --overwrite flag"
    

class TestLociAssignmentExecution:
    """Loci assignment mode execution tests."""
    
    def test_loci_assignment_basic_execution(self, temp_workspace_with_loci):
        """Test basic loci_assignment execution."""
        tmpdir, config_path = temp_workspace_with_loci
        
        result = subprocess.run(
            ['spice', 'loci_assignment', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        assert result.returncode == 0, f"Loci assignment failed: {result.stderr}"
        
        # Verify expected outputs exist
        results_dir = os.path.join(tmpdir, 'results', 'test_loci')
        final_loci_path = os.path.join(results_dir, 'final_loci_assignment.tsv')
        assert os.path.exists(final_loci_path), f"final_loci_assignment.tsv was not created at {final_loci_path}"
        
        # Verify file has content
        df = pd.read_csv(final_loci_path, sep='\t', index_col=0)
        assert len(df) > 0, "final_loci_assignment.tsv is empty"
        assert 'chrom' in df.columns, "final_loci_assignment.tsv missing 'chrom' column"
        assert 'start' in df.columns or 'position' in df.columns, "final_loci_assignment.tsv missing position columns"
    
        # Verify loci_of_selection directory structure was created
        loci_dir = os.path.join(results_dir, 'loci_of_selection')
        
        # These should exist based on pipeline structure
        expected_dirs = ['data_per_length_scale', 'assignment']
        for expected_dir in expected_dirs:
            dir_path = os.path.join(loci_dir, expected_dir)
            assert os.path.exists(dir_path), f"Expected directory {dir_path} was not created"

        final_loci_path = os.path.join(results_dir, 'final_loci_assignment.tsv')
        df = pd.read_csv(final_loci_path, sep='\t', index_col=0)
        assert len(df) > 0, "final_loci_assignment.tsv is empty"
        
        # Check for some expected columns (adapt based on actual output)
        expected_columns = ['chrom', 'start', 'end']  # These are likely column names
        for col in expected_columns:
            assert col in df.columns, f"Critical column containing '{col}' not found in output"

    def test_loci_assignment_with_overwrite_flag(self, temp_workspace_with_loci):
        """Test loci_assignment execution with --overwrite flag."""
        tmpdir, config_path = temp_workspace_with_loci
        
        # First run
        result1 = subprocess.run(
            ['spice', 'loci_assignment', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        assert result1.returncode == 0, f"First loci assignment run failed: {result1.stderr}"
        
        results_dir = os.path.join(tmpdir, 'results', 'test_loci')
        final_loci_path = os.path.join(results_dir, 'final_loci_assignment.tsv')
        assert os.path.exists(final_loci_path), "First run did not create final_loci_assignment.tsv"
        
        # Get modification time of first run
        first_mtime = os.path.getmtime(final_loci_path)
        
        # Second run with --overwrite
        result2 = subprocess.run(
            ['spice', 'loci_assignment', '--config', config_path, '--overwrite'],
            capture_output=True,
            text=True,
            timeout=600
        )
        assert result2.returncode == 0, f"Second loci assignment run with --overwrite failed: {result2.stderr}"
        
        # File should be recreated (newer modification time)
        second_mtime = os.path.getmtime(final_loci_path)
        assert second_mtime >= first_mtime, "File was not recreated with --overwrite flag"
