import yaml
import os
import pdb

def comment():
    import yaml

    class MyDumper( yaml.Dumper ) :
        def __init__ ( self , *args , **kwargs ) :
            self.comments = kwargs.pop( 'comments' , {} )
            super( MyDumper , self ).__init__( *args , **kwargs )

        def increase_indent ( self , flow = False , indentless = False ) :
            return super( MyDumper , self ).increase_indent( flow , False )

        def write_line_break ( self , data = None ) :
            super( MyDumper , self ).write_line_break( data )
            current_indent = len( self.indents )
            if current_indent in self.comments :
                comment = self.comments[current_indent]
                super( MyDumper , self ).write_comment( comment )

    data = {
        'foo' : 'bar' ,
        'baz' : 'qux' ,
    }

    comments = {
        1 : 'This is a comment for baz' ,
        2 : 'This is a comment for foo' ,
    }

    # Dump the dictionary to a YAML file with comments
    with open( 'data.yaml' , 'w' ) as f :
        yaml.dump( data , f , MyDumper , default_flow_style = False , comments = comments )

def distinguish_flat_fielded_3D(image_base_dirs):
    dirs_3D =image_base_dirs[0]
    dirs_flat_fielded=image_base_dirs[1]
    if 'flat' in image_base_dirs[0] :
        dirs_flat_fielded = image_base_dirs[0]
        dirs_3D= image_base_dirs[1]
    if 'flat' in image_base_dirs[1] :
        dirs_flat_fielded = image_base_dirs[1]
        dirs_3D= image_base_dirs[0]
    return dirs_3D ,dirs_flat_fielded

def find_dir(directory,word,base=False):
    tff_dirs = []
    for root , dirs , files in os.walk( directory) :
        # Check each file for a .tff extension and add its directory name to the list if found
        for file in files :
            if word in file :
                if base:
                    if file.endswith(word):
                        tff_dir = os.path.join( root , file )
                    else:
                        tff_dir=[]
                else:
                    tff_dir = os.path.dirname( os.path.join( root , file ) )
                tff_dirs.append( tff_dir )
    try:
        result = list( set( [f for f in tff_dirs] ) )
        return result
    except:
        return None
def main():
    # Define the directory to search
    directory = os.getcwd( )

    # Get a list of all files in the directory
    files = os.listdir( directory )

    # Filter the list of files to include only .tff image files
    image_files =find_dir(directory, '.tif' )

    expt_files = find_dir(directory,  '.expt',base=True )
    refl_files = find_dir(directory, '.refl' ,base = True)
    npy_files = find_dir( directory , '.npy',base = True )
    try:
        refl_file=refl_files[0]
    except:
        refl_file=''
    try:
        expt_file=expt_files[0]
    except:
        expt_file=''
    try:
        npy_file=npy_files [0]
    except:
        npy_file=''
    # Get the directory paths for the image files


    #make the dirs list into a dictionary for creating yaml file
    # expt_my_dict = {i : expt_files[i] for i in range( len( expt_files ) )}
    # refl_my_dict = {i : refl_files[i] for i in range( len( refl_files ) )}
    try:
        dirs_3D , dirs_flat_fielded=distinguish_flat_fielded_3D(image_files)
    except:
        dirs_3D , dirs_flat_fielded='',''
    images_my_dict = {'3d model image directory' : dirs_3D,
                      'flat fielded image directory':dirs_flat_fielded}
    pre_data = {
        'store_dir': directory,
        'dataset': 'test',
        'segimg_path' : dirs_3D,
        'rawimg-path':dirs_flat_fielded,
        'refl_filename':refl_file,
        'expt_filename': expt_file,
        'create3D': True,
        'cal_coefficient': True,
        'coefficient_auto_orientation':False,
        'coefficient_auto_viewing':True,
        'coefficient_orientation': 0,
        'coefficient_viewing': 0,
        'flat_field_name' : None ,
        'coefficient_thresholding':'mean',
        'dials_dependancy':'source /dls/science/groups/i23/yishun/dials_yishun/dials' ,
        'full_reflection': False,
        'model_storepath':npy_file,

    }
    mp_data = {
        'store_dir': directory,
        'liac' : 0 ,
        'loac' : 0 ,
        'crac' : 0 ,
        'buac' : 0 ,
        'num_cores' : 20 ,
        'hour' : 3,
        'minute' : 10 ,
        'second' : 10 ,
        'sampling' : 5000 ,
        'dials_dependancy' : 'source /dls/science/groups/i23/yishun/dials_yishun/dials' ,
        'hpc_dependancies' : 'module load global/cluster' ,
        'dataset': 'test',
        'offset':0,
        'refl_filename':refl_file,
        'expt_filename': expt_file,
        'model_storepath': '',
        'post_process': False,
        'full_reflection' : False ,
        'with_scaling' : True ,
        'anomalous':True,
        'mtz2sca_dependancy' : 'module load ccp4' ,
    }
    post_data = {
        'store_dir': directory,
        'dials_dependancy' : 'source /dls/science/groups/i23/yishun/dials_yishun/dials' ,
        'mtz2sca_dependancy' : 'module load ccp4' ,
        'dataset': 'test',
        'refl_filename':refl_file,
        'expt_filename': expt_file,
        'full_reflection' : False ,
        'with_scaling':True,
    }
    multipost_data = {
        'store_dir': directory,
        'dials_dependancy' : 'source /dls/science/groups/i23/yishun/dials_yishun/dials' ,
        'mtz2sca_dependancy' : 'module load ccp4' ,
        'dataset': 'multi_dataset',
        'refl_filename':refl_files,
        'expt_filename': expt_files,
        'full_reflection' : False ,
        'with_scaling':True,
    }
    # Write the image file paths to a YAML file
    with open( 'default_preprocess_input.yaml' , 'w' ) as file :
        yaml.dump( pre_data , file, default_flow_style=False, sort_keys=False, indent=4 )

    with open( 'default_mpprocess_input.yaml' , 'w' ) as file :
        yaml.dump( mp_data , file, default_flow_style=False, sort_keys=False, indent=4)

    with open( 'default_postprocess_input.yaml' , 'w' ) as file :
        yaml.dump( post_data , file, default_flow_style=False, sort_keys=False, indent=4 )
    with open( 'default_mutli_postprocess_input.yaml' , 'w' ) as file :
        yaml.dump( multipost_data , file,  sort_keys=False, indent=4 )
if __name__ == '__main__':
    main()
