#!/bin/bash
# This script installs the LaTeX packages required for the thesis compilation,
# based on the packages listed in Sections/01_class_and_packages.tex.

set -e

echo "Updating package lists..."
sudo apt-get update

# A list of TeX Live packages that should cover the dependencies in your .tex file.
# We group them to minimize the number of apt-get calls.
TEXLIVE_PACKAGES="
texlive-latex-recommended
texlive-latex-extra
texlive-science
texlive-fonts-recommended
"

echo "Installing required TeX Live packages..."
sudo apt-get install -y $TEXLIVE_PACKAGES

echo "All identified LaTeX packages should now be installed."