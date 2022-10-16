'''
    Ana Dobrianova
    BioInformatics
    This is the source code for the data generated in Assignment 2.
'''

import math
'''
    Divides the genome into num_segments segments of equal length and returns of a list of strings,
    where each string is 1 segment.
    
    Arguments:
        num_segments - int, the number of segments to divide the genome into
        genome - string, the full genome
        
    Returns:
        A list of num_segments strings where each string is a segment of the genome
'''

def segment_genome(num_segments, genome):
    len_segment = int(len(genome) / num_segments)
    print("Genome of length ", len(genome), " will be divided into ", num_segments, " segments of length ", len_segment)
    segmented_genome = []
    
    segment_start = 0
    for i in range(num_segments):
        segmented_genome.append(genome[segment_start:segment_start + len_segment])
        segment_start += len_segment

    return segmented_genome

'''
    Get the k-mers from all the segments
    
    Arguments:
        segmented_genome - a list of strings, where each string is 1 segment
        k_list - a list of ints, where each element is a k
        
    Returns:
        A dictionary where the key is the segment number and the value is a k-d
        array where each index is k - 1 and stores an array of k-mers
'''
def read_k_mers_from_segments(segmented_genome, k_list):
    
    # make a dictionary where each key is a length k and value will be a list of k-mers
    segments_k_mers = {}
    for i in range(len(segmented_genome)):
        segments_k_mers.setdefault(str(i), [])
        
    # fill the dictionary with all the k-mers for each segment
    for seg_num in range(len(segmented_genome)):
        segment = segmented_genome[seg_num]
        for k in k_list:
            segments_k_mers[str(seg_num)].append(read_k_mers(segment, k))

    return segments_k_mers

'''
    Read and return all the k_mers of length k in the given segment
    
    Arguments:
        segment - string, a segment of the genome
        k - int, length of the k-mers to be found in the segment
        
    Returns:
        A list of k-mers of length k in the segment
'''
def read_k_mers(segment, k):
    k_mers = []
    for i in range(len(segment) - k + 1):
        k_mers.append(segment[i:i + k])
    return k_mers

'''
    Count the frequency of a of k_mer in a segment
    
    Arguments:
        seg_k_mer_list - a list of k-mers of one length k from one segment
        k_mer - a string, the k-mer to count in the given list
        
    Returns:
        An int, the number of occurences of that k-mer in that segment
'''
def count_frequency(seg_k_mer_list, k_mer):
    freq = 0
    # count the instances of k_mer in the list
    for item in seg_k_mer_list:
        if item == k_mer:
            freq += 1
    
    return freq

'''
    Find the reverse complement of a k-mer
    
    Arguments:
        k_mer - string, a k-mer
        
    Returns:
        A string, the reverse complement of the k-mer
'''
def find_reverse_complement(k_mer):
    # find the complement of a given k_mer
    rev_comp = []
    for i in k_mer:
        if i == 'a' or i == 'A':
            rev_comp.append('T')
        elif i == 't' or i == 'T':
            rev_comp.append('A')
        elif i == 'c' or i == 'C':
            rev_comp.append('G')
        elif i == 'g' or i == 'G':
            rev_comp.append('C')
    rev_comp.reverse()
    return ''.join(rev_comp)

'''
    Calculate the skew of a k-mer
    
    Arguments:
        k_mer - string, the k-mer to find the skew of
        comp - string, the reverse complement of the k_mer
        
    Returns:
        A float, the skew of the k-mer
'''
def calculate_skew(k_mer_freq, comp_freq):
    # calculate the skew for a given k-mer m
    if k_mer_freq == 0:
        k_mer_freq = 1
    if comp_freq == 0:
        comp_freq = 1
    return math.log((k_mer_freq / comp_freq))


'''
    Build a table for the frequencies of k-mers of length k
    
    Arguments:
        k_mer_dict - a dictionary, key is segment number and value is k-D array of k-mers
        k - an int, length of the k-mers to make the table for

    Returns:
        An 4**k x num_segments matrix of tuples, where a tuple is 

'''
def build_frequency_table(k_mer_dict, k):
    # create empty table
    table = {}
    
    # sort k-mers in dictonary
    for segment in k_mer_dict.keys():
        sorted_k_mers =  sorted(k_mer_dict[segment][k - 1])
        k_mer_dict[segment][k - 1] = sorted_k_mers
    num_segments = int(list(k_mer_dict.keys())[-1]) + 1
    
    # traverse the list of k-mers for each segment, counting the 
    #   frequencies of the k-mers
    for segment in k_mer_dict.keys():
        
        # get the list of k_mers for that segment
        segment_list = k_mer_dict[segment][k - 1]
        # SEGMENT NUMBERS START AT 0
        '''
            All cases for a k_mer:
                1. not in table and doesn't exist in the segment
                2. not in in table and is in segment
                3. in table and doesn't exist in the segment
                4. in table, is in segment, and already recorded for this segment
                5. in table, is in segment, and hasn't been recorded for this segment
        '''
        for k_mer in segment_list:
            
            # k-mer already in the table and it is in this current segment
            if k_mer in table.keys() and find_k_mer_in_segment(k_mer_dict, k_mer, k, segment): # 4 or 5   
                # check if the length of table[k_mer] is less than the segment number, 
                if table[k_mer][-1][0] == int(segment):
                    continue
                else: 
                    freq = count_frequency(segment_list, k_mer)
                    table[k_mer].append((int(segment), freq))
            # k-mer is not in the table and is in the current segment
            elif (k_mer not in table.keys()) and find_k_mer_in_segment(k_mer_dict, k_mer, k, segment): # 2
                table.setdefault(k_mer, [])
                freq = count_frequency(segment_list, k_mer)
                table[k_mer].append((int(segment), freq))
                
    for k_mer in table.keys(): # go through the whole thing again and make sure each segment has a spot in each array
        i = 0
        if table[k_mer][0][0] != 0:
                table[k_mer].insert(0, (0, 0))
        while int(table[k_mer][-1][0]) != num_segments - 1:
            table[k_mer].append((int(table[k_mer][-1][0]) + 1, 0))
        while len(table[k_mer]) != num_segments:  
            if table[k_mer][i][0] + 1 != table[k_mer][i + 1][0]:
                table[k_mer].insert(i + 1, (i + 1, 0))
            i += 1

    return table

'''
    Find a k-mer in the table and return the value(s) associated with it
    
    Arguments:
        table - a dictionary of lists, full of k-mer:[values]
        k_mer - a string, the k-mer to find in the table
    
    Returns:
        The values associated with the k-mer
        If the k-mer is not found in the table, it returns -1

'''
def find_k_mer_in_table(table, k_mer, seg_num):
    try:
        val = table[k_mer][seg_num][1]
    except:
        val = 1
    return val

'''
    Find a k-mer in the k-mer dictionary organized by segment numbers
    
    Arguments:
        k_mer_dict - a dictionary of "segment_number":k-D array of k-mers
        k_mer - a string, the k-mer to find in the dictionary
        k - int, length of k-mer
        seg_num - int, the segment number to look in
        
    Returns:
        True if it's in the segment, False otherwise
'''
def find_k_mer_in_segment(k_mer_dict, k_mer, k, seg_num):
    if k_mer in k_mer_dict[seg_num][k - 1]:
        return True
    
    return False


'''
    Take the frequency table and turn it into a table of the skews of the k-mers
    
    Arguments:
        freq_table - a dictionary of k_mer:[(segment_number, frequencies)]
        
    Returns
        the same table but with skews instead of frequencies
'''
def build_skew_table(freq_table):
    skew_table = {}
    for k_mer in freq_table.keys():
        skew_table.setdefault(k_mer, [])
        seg_num = 0
        for item in freq_table[k_mer]:
            k_mer_comp = find_reverse_complement(k_mer)
            k_mer_freq = find_k_mer_in_table(freq_table, k_mer, seg_num)
            k_mer_comp_freq = find_k_mer_in_table(freq_table, k_mer_comp, seg_num)
            skew = calculate_skew(k_mer_freq, k_mer_comp_freq)
            skew_table[k_mer].append(skew)
            seg_num += 1

    return skew_table

'''
    Print the input table,  correctly formatted
    
    Arguments:
        table - a dictionary, the table to print
    Returns:
        Nothing
'''
def print_table(table, num_segs):
    print(' '.join(['{:6d}'.format(i) for i in range(1, num_segs + 1)]))
    for k, v in table.items():
        if type(v[0]) == tuple:
            vals_sig = [round(i[1], 3) for i in v]
        else:
            vals_sig = [round(i, 3) for i in v]
        print(k, end=" ")
        print('  '.join(['{: =.3f}'.format(i) for i in vals_sig]))

if __name__=="__main__":
    # open file with ecoli genome and read it
    file = open("sequence.fasta", "r")
    ecoli_full = str(file.read())

    # divide the genome into equal lengths
    num_segments = 50
    k = 3
    ecoli_segments = segment_genome(num_segments, ecoli_full)
    # get all the k-mers of lengths 1 to 3 for each segment
    k_list = range(1, 4)
    k_mer_dict = read_k_mers_from_segments(ecoli_segments, k_list)
    
    # create a table of frequencies of each 1-mer 
    k = 1
    freq_table_1 = build_frequency_table(k_mer_dict, k)
    skew_table_1 = build_skew_table(freq_table_1)
    print("SKEWS k = 1")
    print_table(skew_table_1, num_segments)
    
    # create a table of frequencies of each 2-mer 
    print()
    k = 2
    freq_table_2 = build_frequency_table(k_mer_dict, k)
    skew_table_2 = build_skew_table(freq_table_2)
    print("SKEWS k = 2")
    print_table(skew_table_2, num_segments)
    
    # create a table of frequencies of each 2-mer 
    print()
    k = 3
    freq_table_3 = build_frequency_table(k_mer_dict, k)
    skew_table_3 = build_skew_table(freq_table_3)
    print("SKEWS k = 3")
    print_table(skew_table_3, num_segments)
    