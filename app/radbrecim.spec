# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('src/assets', 'src/assets'),
        ('src/processing/classifier_best.pth', 'src/processing'),
        ('src/processing/run32best-checkpoint-epoch=36-val_acc=0.85.ckpt', 'src/processing'),
        ('../src/models/nn_inference.py', 'src/models'),
        ('../src/models/cl_inference.py', 'src/models'),
        ('../src/models/config/cl_models/tear_classifier.pkl', 'src/models/config/cl_models'),
    ],
    hiddenimports=[
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'matplotlib.backends.backend_qtagg',
        'sklearn.ensemble._forest',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.neighbors._partition_nodes',
        'torch', 'torchvision', 'torchvision.models',
        'lightning', 'timm', 'albumentations',
        'skimage.feature', 'skimage.morphology',
        'cv2', 'PIL',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RadBrecim',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon='src/assets/images/logo.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='RadBrecim',
)
