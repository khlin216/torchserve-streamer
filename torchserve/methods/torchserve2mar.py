# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

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
        coordinator: str,
        model_version: str = '1.0',
        force: bool = False,
):
    """Converts MMDetection model (config + checkpoint) to TorchServe `.mar`.
    """
    mmcv.mkdir_or_exist(output_folder)
    dummy_file = "methods/constants.py"
    if os.path.isfile("./serve_alldet/all_det.mar"):
        os.remove("./serve_alldet/all_det.mar")
    with TemporaryDirectory() as tmpdir:
        # shutil.copytree("./", tmpdir, dirs_exist_ok=True)
        for source in os.listdir("./"):
            if os.path.isdir(os.path.join("./", source)):
                shutil.copytree(os.path.join("./", source), os.path.join(tmpdir, source))
            else:
                shutil.copyfile(os.path.join("./", source), os.path.join(tmpdir, source))
        shutil.move(os.path.join(tmpdir, coordinator), "coordinator.py")
        print(tmpdir)
        print(os.listdir(src_root))
        args = Namespace(
            **{
                'model_file': dummy_file,
                'handler': "coordinator.py",
                'model_name': model_name,
                'model_path' : tmpdir,
                'version': model_version,
                'export_path': output_folder,
                "serialized_file": dummy_file,
                'requirements_file': dummy_file,
                'force': force,
                'extra_files' : tmpdir,
                'runtime': 'python',
                'archive_format': 'default',
                'convert': False
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


if __name__ == '__main__':
    # mmdet2torchserve("./", "./serve_alldet/", "mtcnn", "./model_files",coordinator="./coordinators/coordinator.py", force=True) # dont add unless its needed as torchserve will automatically host it
    mmdet2torchserve("./", "./serve_alldet/", "triangle", "./model_files",coordinator="./coordinators/triangles_coordinator.py", force=True)
