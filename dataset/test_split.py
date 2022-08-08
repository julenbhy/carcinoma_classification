#!/usr/bin/python3

# import OS module
import os
import random


split = 0.2

### NEC filse ###

# Get the list of all files and directories
origin_path = "./train/NEC/"
dest_path = "./test/NEC/"

dir_list = os.listdir(origin_path)

total=len(dir_list)
test_files= int(total*split)

print("NEC: Detected ", total, "images, moving ", test_files, "to test directory")


random.shuffle(dir_list)
for i in range(test_files):
	#print('mv ' + origin_path + dir_list[i] + ' ' + dest_path)
	os.system('mv ' + '"' + origin_path + dir_list[i]+ '"'  + ' ' + dest_path)



### NEC filse ###

# Get the list of all files and directories
origin_path = "./train/NET1/"
dest_path = "./test/NET1/"

dir_list = os.listdir(origin_path)

total=len(dir_list)
test_files= int(total*split)

print("NET1: Detected ", total, "images, moving ", test_files, "to test directory")


random.shuffle(dir_list)
for i in range(test_files):
        #print('mv ' + origin_path + dir_list[i] + ' ' + dest_path)
        os.system('mv ' + '"' + origin_path + dir_list[i]+ '"'  + ' ' + dest_path)

