#!/bin/bash

echo "Installing Chinese fonts for matplotlib..."
echo "This script will install Chinese fonts to enable proper display of Chinese characters."

# Update package list
sudo apt update

# Install Chinese fonts
echo "Installing Noto CJK fonts..."
sudo apt install -y fonts-noto-cjk

echo "Installing WenQuanYi fonts..."
sudo apt install -y fonts-wqy-microhei

echo "Installing additional Chinese fonts..."
sudo apt install -y fonts-wqy-zenhei fonts-arphic-uming

# Refresh font cache
echo "Refreshing font cache..."
sudo fc-cache -fv

echo ""
echo "Chinese fonts installation complete!"
echo ""
echo "To enable Chinese language in the plot:"
echo "1. Open plot/bug_status_plot.py"
echo "2. Change LANGUAGE = \"en\" to LANGUAGE = \"zh\""
echo "3. Run: python plot/bug_status_plot.py"
echo ""
echo "The script will now automatically detect and use Chinese fonts."
