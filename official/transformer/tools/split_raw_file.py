#/usr/bin/env python
#coding=UTF-8
import logging
import sys

from optparse import OptionParser

# Configuration of options.
usage = 'usage: %prog [options]'
parser = OptionParser(usage=usage)
parser.add_option('-i', '--input_file', dest='input_file',
    help='Input file.', metavar='FILE')
parser.add_option('-c', '--src_ctx_file', dest='source_context_file',
    help='Output file.', metavar='FILE')
parser.add_option('-d', '--tgt_ctx_file', dest='target_context_file',
    help='Output file.', metavar='FILE')
parser.add_option('-v', '--src_voc_file', dest='source_vocab_file',
    help='Output file.', metavar='FILE')
parser.add_option('-w', '--tgt_voc_file', dest='target_vocab_file',
    help='Output file.', metavar='FILE')

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

def main():
    options, args = parser.parse_args()
    input_file = options.input_file
    source_context_file = options.source_context_file
    target_context_file = options.target_context_file
    source_vocab_file = options.source_vocab_file
    target_vocab_file = options.target_vocab_file
    with open(input_file) as fin, open(source_context_file, 'w') as fsc, open(target_context_file, 'w') as ftc, open(source_vocab_file, 'w') as fsv, open(target_vocab_file, 'w') as ftv:
        for line in fin:
            line = line.strip('\n')
            try:
              _, _, source_line, target_line = line.split('\t')
            except:
              continue

            source_items = source_line.split('\1')
            target_items = target_line.split('\1')
            if len(source_items) != 3 or len(target_items) != 3:
              continue

            fsc.write('%s\n' % source_line)
            fsv.write('%s\n' % source_items[1])
            ftc.write('%s\n' % target_line)
            ftv.write('%s\n' % target_items[1])


if __name__ == '__main__':
    main()
