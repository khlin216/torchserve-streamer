# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import mmcv

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    raise(ImportError)


def mmdet2torchserve(
        src_root: str,
        output_folder: str,
        model_name: str,
        model_path: str,
        model_version: str = '1.0',
        force: bool = False,
):
    """Converts MMDetection model (config + checkpoint) to TorchServe `.mar`.
    """
    mmcv.mkdir_or_exist(output_folder)
    dummy_file = "methods/README.md"
    if os.path.isfile("./serve_alldet/all_det.mar"):
        os.remove("./serve_alldet/all_det.mar")
    with TemporaryDirectory() as tmpdir:
        print(os.listdir(src_root))
        args = Namespace(
            **{
                'model_file': dummy_file,
                'handler': f'coordinator.py',
                'model_name': model_name,
                'model_path' : "./",
                'version': model_version,
                'export_path': output_folder,
                "serialized_file": dummy_file,
                'requirements_file': dummy_file,
                "extra_files": "./",
                'force': force,
                'runtime': 'python',
                'archive_format': 'default',
                'convert': False
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


if __name__ == '__main__':
    mmdet2torchserve("./", "./serve_alldet/", "all_det", "./model_files", force=True)
