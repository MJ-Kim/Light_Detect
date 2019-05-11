import os, sys
from PIL import Image

width = 720
height = 360

def resize(filename, savename):
    global width, height
    
    image = Image.open(filename)
    resize_image = image.resize((width, height))
    resize_image.save(savename)


def search(dirname):
    filename_list = os.listdir(dirname)
    length = len(filename_list) - 1
    
    filenames = enumerate(filename_list)
    
    for i, filename in filenames:
        if i == length:
            break
        
        fullFileName = os.path.join(dirname, filename)
        ext = os.path.splitext(fullFileName)[-1]

        savedir = os.path.join(dirname, 'resized')
        savename =  os.path.join(savedir, filename)
        fullpath = os.path.join(filename, savename)

        targetfile = os.path.join(dirname, filename)

        if ext == '.jpg':
            print('processing...', (i+1), '/', length, '(' + fullpath + ')')
            resize(targetfile, savename)


def makedir(target_dirname):
    save_dir = os.path.join(target_dirname, 'resized')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)



if __name__ == "__main__" :
    target_dirname = sys.argv[1]
    width, height = int(sys.argv[2]), int(sys.argv[3])
    
    makedir(target_dirname)
    search(target_dirname)




