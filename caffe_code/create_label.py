import os
def create_image_list(file_path,txtpath,num):


    image_name_list=os.listdir(file_path)

    with open(txtpath,'a') as f:
        print('saving to'+txtpath+'...')
        for image_name in image_name_list:
            image_label=str(num)
            image_data='trainingSet/'+str(num)+'/'+image_name+' '+image_label+'\n'
            f.write(image_data)
        print('done')

if __name__=="__main__":
    set_path="/home/assassin/dataset/MNIST/trainingSet/"
    txtpath="/home/assassin/dataset/MNIST/trainingSet/train.txt"
    if os.path.isfile(txtpath):
        os.remove(txtpath)

    for i in range(0,10):
        file_path=set_path+str(i)+'/'
        create_image_list(file_path,txtpath,i)
    