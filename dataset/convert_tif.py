from PIL import Image
import os
import glob
import sys
Image.MAX_IMAGE_PIXELS = None


DOWNSAMPLE = 50  # The downsample applied to the images
#SOURCE = './database/NEC/'  # The source path (will search in subdirectories)
#DEST = './database/NEC_IMG/'  # The destination path

if(len(sys.argv) != 3):
    print('ERROR: incorrect number of arguments')
    print('\t arg1: source directory to get the .tif images (will search in subdirectories)')
    print('\t arg2: destination directory to save the .jpg images')
    sys.exit(1)

SOURCE = sys.argv[1]+'/'
DEST = sys.argv[2]+'/'

# create DEST directory if doesn't exist
if not os.path.exists(DEST):
    os.makedirs(DEST)
    print('Created', DEST, 'directory')

# Find .tif images reculsively
images = glob.glob(SOURCE + "/**/*.tif", recursive = True)
for img in images:
    print('Converting: ', img)

    # Generate new name
    target_name = aux_name = DEST + os.path.splitext(img)[0].split('/')[-1]
    id = 1
    while os.path.exists(target_name + '.jpg'):  # In case filename exists
        target_name = aux_name+'_'+str(id)
        id += 1
        
    i = Image.open(img)  # Open, downsample and convert image
    w,h = i.size
    new_img = i.resize((int(w/DOWNSAMPLE), int(h/DOWNSAMPLE)))
    rgb_image = new_img.convert('RGB')
    
    print('Saved: ', target_name,'.jpg')
    rgb_image.save(target_name+'.jpg')
