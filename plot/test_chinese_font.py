#!/usr/bin/env python3
"""
Simple test script to verify Chinese font support in matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Rebuild font cache
fm.fontManager.__init__()

# Set up Chinese font
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif'],
    'axes.unicode_minus': False
})

# Create a simple test plot
fig, ax = plt.subplots(figsize=(8, 6))

# Test Chinese text
chinese_text = ["未确认", "已确认且正在修复", "已确认且已修复"]
values = [30, 40, 30]

ax.bar(range(len(chinese_text)), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_xticks(range(len(chinese_text)))
ax.set_xticklabels(chinese_text)
ax.set_ylabel('百分比 (%)')
ax.set_title('中文字体测试')

plt.tight_layout()
plt.savefig('chinese_font_test.png', dpi=150, bbox_inches='tight')
print("Chinese font test completed! Check 'chinese_font_test.png'")
print("If Chinese characters display correctly, the font setup is working.")
