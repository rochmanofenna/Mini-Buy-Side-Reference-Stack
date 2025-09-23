from setuptools import setup, find_packages

setup(
    name="ai-trading-infra",
    version="0.1.0",
    description="Production-style AI trading infrastructure reference stack",
    author="AI Trading Infrastructure Contributors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.109.0",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "black>=23.12.1",
            "ruff>=0.1.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "backtest=src.strategy.backtest_runner:main",
            "router=src.router.main:main",
        ],
    },
)