import yaml
import os
import pdb
import sys
parent_dir =os.path.dirname( os.path.abspath(__file__))

sys.path.append(parent_dir)
# pdb.set_trace()
from param import set_parser_init


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
    args = set_parser_init( )
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
    if args.pre:
        pre_data = {
            'store_dir': directory,
            'dataset': 'test',
            'segimg_path' : dirs_3D,
            'rawimg-path':dirs_flat_fielded,
            'refl-path':refl_file,
            'expt-path': expt_file,
            'create3D': True,
            'cal_coefficient': True,
            'coefficient_auto_orientation':False,
            'coefficient_auto_viewing':True,
            'coefficient_orientation': 0,
            'coefficient_viewing': 0,
            'flat_field_name' : None ,
            'coefficient_thresholding':'mean',
            'dials_dependancy':'module load dials/latest' ,
            'full_reflection': False,
            'model_storepath':npy_file,

        }
        # Dump the dictionary to a YAML string
        yaml_str = yaml.dump(pre_data, default_flow_style=False, sort_keys=False, indent=4)

        # Define the line where the comment should be added
        comment = "    # if flat-fielded thresholding doesn't look good, can try 'otsu', or 'li' or 'yen'  \n"
        comment2="# examples of effect can be shown on https://imagej.net/plugins/auto-threshold \n"
        key_to_comment = 'coefficient_thresholding:'

        # Insert the comment after the specified key
        lines = yaml_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(key_to_comment):
                # Insert the comment after the current line
                lines.insert(i + 1, comment.strip())
                lines.insert(i + 2, comment2.strip())
                break  # Assuming only one match, we can break after inserting the comment

        # Join the lines back together
        yaml_str_with_comment = '\n'.join(lines)

        # Write the modified YAML string to a file
        with open('default_preprocess_input.yaml', 'w') as file:
            file.write(yaml_str_with_comment)

        print('preprocess input file created')
    if args.mp:   
        mp_data = {
            'store_dir': directory,
            'dataset': 'test',
            'liac' : 0 ,
            'loac' : 0 ,
            'crac' : 0 ,
            'buac' : 0 ,
            'num_cores' : 20 ,
            'hour' : 6,
            'minute' : 10 ,
            'second' : 10 ,
            'sampling_ratio' : 0.1 ,
            'dials_dependancy' : 'module load dials/latest' ,
            'hpc_dependancies' : 'module load global/cluster' ,
            'offset':0,
            'refl_path':refl_file,
            'expt_path': expt_file,
            'model_storepath': '',
            'post_process': True,
            'full_reflection' : 0 ,
            'with_scaling' : True ,
            'anomalous':False,
            'mtz2sca_dependancy' : 'module load ccp4' ,
        }
        with open( 'default_mpprocess_input.yaml' , 'w' ) as file :
            yaml.dump( mp_data , file, default_flow_style=False, sort_keys=False, indent=4)

    if args.post:
        post_data = {
            'store_dir': directory,
            'dials_dependancy' : 'source /dls/science/groups/i23/yishun/dials_yishun/dials' ,
            'mtz2sca_dependancy' : 'module load ccp4' ,
            'dataset': 'test',
            'refl_path':refl_file,
            'expt_path': expt_file,
            'full_reflection' : 0,
            'with_scaling':True,
        }


        with open( 'default_postprocess_input.yaml' , 'w' ) as file :
            yaml.dump( post_data , file, default_flow_style=False, sort_keys=False, indent=4 )

if __name__ == '__main__':
    main()
