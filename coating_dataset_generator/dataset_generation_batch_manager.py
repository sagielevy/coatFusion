import logging
import os
import sys
import time
import argparse
from pathlib import Path

import bpy

from .config import get_default_config
from .dataset_orchestrator import generate_dataset


def setup_logging(job_id):
    """Set up logging for this batch job."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Job %(job_id)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"batch_process_{job_id}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Add job_id to all log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.job_id = job_id
        return record

    logging.setLogRecordFactory(record_factory)


def load_project(config): 
    """Load a Blender project file"""
    if config.benchmark_mode:
        blend_file_path = r"coating_dataset_generator/Benchmarks/coating_dataset_generation_benchmark.blend"
    else:
        blend_file_path = r"coating_dataset_generator/coating_dataset_generation.blend"
    
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    print(f"Loaded Blender project: {blend_file_path}")


def configure_gpu_device_type(logger, device_type='AUTO'):
    """
    Configure specific GPU device type

    Args:
        device_type: 'AUTO', 'CUDA', 'OPTIX', 'OPENCL', 'HIP', 'METAL'
    """

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    if device_type == 'AUTO':
        # Auto-detect best GPU type
        cycles_prefs.get_devices()

        # Priority order: OPTIX > CUDA > HIP > METAL > OPENCL
        preferred_types = ['OPTIX', 'CUDA', 'HIP', 'METAL', 'OPENCL']

        for pref_type in preferred_types:
            for device in cycles_prefs.devices:
                if device.type == pref_type:
                    cycles_prefs.compute_device_type = pref_type
                    print(f"Auto-selected GPU type: {pref_type}")
                    return pref_type

        logger.error("No GPU found, will use CPU")
        return 'CPU'
    else:
        cycles_prefs.compute_device_type = device_type
        logger.info(f"Set GPU type to: {device_type}")
        return device_type


def main():
    parser = argparse.ArgumentParser(description="Batch process coating dataset generation")
    parser.add_argument("initial_sample_index", type=int, help="Initial sample index")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate")
    parser.add_argument("--indices_list", type=str, help="Comma-separated list of indices to generate")
    parser.add_argument("--names_list", type=str, help="Comma-separated list of object names to generate")

    # If the script is invoked from bash, argv might contain blender args before `--`
    # We only care about the args passed after `python -m ...`
    # Let's use standard parsing but we need to handle old bash scripts that didn't use named arguments.
    # To maintain backward compatibility with old script without argparse:
    
    config = get_default_config()

    try:
        # We need to handle both old positional argument (3rd argument as indices_list) and new ones.
        if len(sys.argv) > 3 and not sys.argv[3].startswith('--'):
            initial_sample_index = int(sys.argv[1])
            num_samples = int(sys.argv[2])
            indices_str = sys.argv[3]
            indices_list = [int(x.strip()) for x in indices_str.split(',')]
            names_list = None
            config.continue_generation = False
        else:
            args = parser.parse_args()
            initial_sample_index = args.initial_sample_index
            num_samples = args.num_samples
            
            indices_list = None
            if args.indices_list:
                indices_list = [int(x.strip()) for x in args.indices_list.split(',')]
                config.continue_generation = False # Must override given indices
                
            names_list = None
            if args.names_list:
                names_list = [x.strip() for x in args.names_list.split(',')]
                config.continue_generation = False # Must override given names
            
    except ValueError:
        print("Error: Arguments must be integers where expected")
        sys.exit(1)

    # Get job ID from SLURM environment or use default
    job_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'unknown')

    setup_logging(job_id)
    logger = logging.getLogger(__name__)

    bpy.context.scene.render.engine = 'CYCLES'
    configure_gpu_device_type(logger)

    logger.info(f"Loading project")
    load_project(config)
    logger.info(f"Project loaded")

    logger.info(f"Starting batch processing")
    
    if indices_list is not None:
        logger.info(f"Using specific indices: {indices_list}")
        logger.info(f"Number of indices: {len(indices_list)}")
    elif names_list is not None:
        logger.info(f"Using specific names: {names_list}")
        logger.info(f"Number of names: {len(names_list)}")
    else:
        logger.info(f"Initial sample index: {initial_sample_index}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Processing samples {initial_sample_index} to {initial_sample_index + num_samples - 1}")

    # Log system information
    logger.info(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    logger.info(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}")

    start_time = time.time()

    generate_dataset(logger, config, initial_sample_index, num_samples, indices_list, names_list)

    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")

    sample_count = len(indices_list) if indices_list is not None else num_samples
    if names_list is not None:
        logger.info(f"Average time per name: {processing_time / len(names_list):.2f} seconds")
    else:
        logger.info(f"Average time per sample: {processing_time / sample_count:.2f} seconds")
    logger.info("Batch processing completed")


if __name__ == "__main__":
    main()