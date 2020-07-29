'''
created on 11/07/2020

@author: Ravindra Kumar
'''

from exp.Required_Modules_And_Packages import *

#function to untar data and unzip agreed label
def untar_data_and_unzip_label(tar_data_file,zip_agreed_label_file):
    path = Path('/notebooks/MultimodalTweetsClassification')
    zf = tarfile.open(path/'data'/tar_data_file)
    zf.extractall(path/'data')
    zf = zipfile.ZipFile(path/'data'/zip_agreed_label_file)
    zf.extractall(path/'data')

# function to remove non-ASCII chars from data
def clean_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

# dictionary of tsv files of all three train,dev, and test split for both task
tsv_data_files ={'humanitarian_task_tsv_files':('data/task_humanitarian_text_img_agreed_lab_train.tsv',
                                                'data/task_humanitarian_text_img_agreed_lab_dev.tsv',
                                                'data/task_humanitarian_text_img_agreed_lab_test.tsv'),
                'Informativeness_task_tsv_files':('data/task_informative_text_img_agreed_lab_train.tsv',
                                                  'data/task_informative_text_img_agreed_lab_dev.tsv',
                                                  'data/task_informative_text_img_agreed_lab_test.tsv')
                }

#function to get tsv files of all three splits for a particular task
def get_tsv_data_files(task_name_tsv_files):
    if task_name_tsv_files == 'Informativeness_task_tsv_files':info=True
    else: info=False
    return list(tsv_data_files[task_name_tsv_files])+[info]

def get_dataframe(train_tsv,dev_tsv,test_tsv,info,path):
    print('reading data and preprocessing it.....')
    train=pd.read_csv(train_tsv,delimiter='\t',encoding='utf-8')
    if info: train = train.drop(0,axis=0)
    dev=pd.read_csv(dev_tsv,delimiter='\t',encoding='utf-8')
    test=pd.read_csv(test_tsv,delimiter='\t',encoding='utf-8')
    
    #to remove redundant links,hashtag sign, and non-ASCII characters 
    train['tweet_text']=train['tweet_text'].apply(lambda x:aidrtokenize.tokenize(x))
    dev['tweet_text']=dev['tweet_text'].apply(lambda x:aidrtokenize.tokenize(x))
    test['tweet_text']=test['tweet_text'].apply(lambda x:aidrtokenize.tokenize(x))

    train['tweet_text'] = train['tweet_text'].apply(lambda x:clean_ascii(x))
    dev['tweet_text'] = dev['tweet_text'].apply(lambda x:clean_ascii(x))
    test['tweet_text'] = test['tweet_text'].apply(lambda x:clean_ascii(x))

    train['is_valid'] = False
    dev['is_valid'] = True


    data = pd.concat([train,dev],axis=0).reset_index()
    data = data.drop(['index'],axis=1)
    test_data = test
    print("done!!")    
    return data,test_data

#function to create databunch object for language model to get encoder
def get_text_data_for_language_model(dataframe,path,batch_size=64):
    text_data_lm = (TextList
               .from_df(dataframe,path,cols='tweet_text')
               #Where are the text? Column tweet_text of data dataframe
               .split_from_df(col='is_valid')
               #How to split it? using column is_valid of dataframe
               .label_for_lm()
               #Label it for a language model
               .databunch(bs=batch_size))
               #Finally we convert to a DataBunch
    return text_data_lm

#function to create databunch object of text data consist of train and dev split
def get_text_data(dataframe,path,pad_first=True,pad_idx=1,batch_size=64,processor=None,vocab=None):  
    text_data = (TextList.from_df(dataframe,path, cols='tweet_text', processor=processor,vocab=vocab)
             .split_from_df(col='is_valid')
             .label_from_df(cols= 'label_text')
             .databunch(bs=batch_size,pad_first=pad_first, pad_idx=pad_idx))
    return text_data

#function to create databunch object of test text data using test split
def get_test_text_data(dataframe,path,pad_first=True,pad_idx=1,batch_size=64,processor=None,vocab=None):
    test_text_data = (TextList.from_df(dataframe,path,cols='tweet_text',processor=processor,vocab=vocab)
             .split_none()
             .label_from_df(cols= 'label_text')
             .databunch(bs=batch_size,pad_first=pad_first, pad_idx=pad_idx))
    return test_text_data

# prepreprocessing and creating image data for classification
def get_image_data(dataframe,path,image_size=224,batch_size=64):
    tfms = get_transforms()
    image_data = (ImageList.from_df(dataframe,path/'data/CrisisMMD_v2.0',cols='image')
            #Where to find the data? -> from dataframe
            .split_from_df(col='is_valid')
            #How to split in train/valid? -> using is_valid column of dataframe
            .label_from_df(cols='label_image')
            #How to label? -> use the label_image column of the dataframe and split
            .transform(tfms, size=image_size)
            #Data augmentation? -> use tfms with a size of image_size(here size of the image 224 have been taken)
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))
            #Finally -> use the imagenet defaults stats for conversion to databunch
    return image_data

# prepreprocessing and creating image test data for classification
def get_test_image_data(dataframe,path,image_size=224,batch_size=64):
    tfms = get_transforms()
    test_image_data = (ImageList.from_df(dataframe,path/'data/CrisisMMD_v2.0',cols='image')
            .split_none()
            .label_from_df(cols='label_image')
            .transform(tfms, size=image_size)
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))
    return test_image_data