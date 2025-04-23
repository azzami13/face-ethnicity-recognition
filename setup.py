from setuptools import setup, find_packages

setup(
    name="face_ethnicity_recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.1",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "tensorflow>=2.6.0",
        "mtcnn>=0.1.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.61.0",
        "Pillow>=8.2.0",
        "streamlit>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A system for face similarity and ethnicity detection",
    keywords="face, recognition, ethnicity, computer vision",
    python_requires=">=3.7",
)