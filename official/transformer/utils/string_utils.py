#coding=utf-8
# Author: Jiahui Dai
# Update: 2019/11/11
import re
import sys

ZHCN_PTN = re.compile('[\u4e00-\u9fa5]+')
SPACE_MARK = '…'

def ispunct(c):
  c = ord(c)
  return  ((c <= 0x002F) or
          (c == 0x00D7) or
          (c == 0x00F7) or
          (c >= 0x003A and c <= 0x003F) or
          (c >= 0x005B and c <= 0x0060) or
          (c >= 0x007B and c <= 0x00BF) or
          (c >= 0x023F and c <= 0x0385) or
          (c >= 0x0510 and c <= 0x10FF) or
          (c >= 0x1200 and c <= 0x1DFF) or
          (c >= 0x2000 and c <= 0x302F) or
          (c >= 0xA000 and c <= 0xABFF) or
          (c >= 0xD7A4 and c <= 0xF8FF) or
          (c >= 0xFE00 and c <= 0xFE6B) or
          (c >= 0xFF01 and c <= 0xFF0F) or
          (c >= 0xFF1A and c <= 0xFF20) or
          (c >= 0xFF3B and c <= 0xFF40) or
          (c >= 0xFF5B and c <= 0xFF64))


def split_zhcn(input_line):
  output_line = ''
  for char in input_line:
    if char == ' ':
      output_line += '%s ' % SPACE_MARK
    elif ispunct(char) or ZHCN_PTN.match(char):
      output_line += ' %s ' % char
    else:
      output_line += char
  output_line = output_line.strip()
  while True:
    if '  ' not in output_line:
      break

    output_line = output_line.replace('  ', ' ')
  return output_line


if __name__ == '__main__':
  a = "你好！小朋友 玩car玩得开心吗o"
  print(split_zhcn(a))
