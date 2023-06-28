import os
import glob
import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True 
import pdb
# Define the path to the root directory containing the folders
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--coefficient-viewing" ,
    type = int ,
    default = 0 ,
    help = "the viewing angle of the 3D model to have the best region to determine absorption coefficient"
            "in degree" ,
)
global args
args = parser.parse_args( )

def pipeline():
    root_directory = '/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/single_datasets/'
    reference_dataset='/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/single_datasets/1/'
    reference_yaml='default_mpprocess_input.yaml'
    term_1='3500ev_1' # specify the term to be replaced in reflection and experiment path
    term_2='x3500ev1' # specify the term to be replaced in reflection and experiment path
    dataset_list=['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                  '16','17','18','19','20','21','22','23','24','25','26']
    
    for dataset in dataset_list:
        try:
            os.makedirs(os.path.join(root_directory,dataset))
            print('Folder {}is  created'.format(dataset))
        except:
            print('Folder {} already exists'.format(dataset))

        with open(os.path.join(reference_dataset,reference_yaml), 'r') as file:
            yaml_data = yaml.load(file)
            
        new_term_1 = term_1.replace('1',dataset)
        new_term_2 = term_2.replace('1',dataset)
        refl = yaml_data['refl_path']
        expt = yaml_data['expt_path']
        refl = refl.replace(term_1,new_term_1)
        refl = refl.replace(term_2,new_term_2)
        expt = expt.replace(term_1,new_term_1)
        expt = expt.replace(term_2,new_term_2)
        assert os.path.isfile(refl) == True
        assert os.path.isfile(expt) == True

        yaml_data['refl_path'] = refl
        yaml_data['expt_path'] = expt
        yaml_data['dataset'] = dataset
        yaml_data['store_dir'] = os.path.join(root_directory,dataset)
        with open(os.path.join(root_directory,dataset,reference_yaml), 'w') as updated_file:
            yaml.dump(yaml_data, updated_file)
        print('new YAML file of dataset {} is created'.format(dataset))
        

def change_orientation_of_yaml_file():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)
    new_dataset_name='orientation_{}'.format(args.coefficient_viewing)
    abs={'coefficient_viewing':args.coefficient_viewing,
        'dataset':new_dataset_name,}
    root_directory = current_file_dir
    old_name='default_preprocess_input.yaml'
    new_name='changed_preprocess_ori_{}.yaml'.format(args.coefficient_viewing)
    yaml_files=os.path.join(root_directory,old_name)

        # Iterate through each YAML file

    with open(yaml_files, 'r') as file:
        yaml_data = yaml.load(file)
        
        for key in abs:
            yaml_data[key]=abs[key]
        
        # Save the updated YAML content
        with open(os.path.join(root_directory,new_name), 'w') as updated_file:
            yaml.dump(yaml_data, updated_file)

def change_contents_of_yaml_file():


    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)

    root_directory = current_file_dir
    old_name='default_mpprocess_input.yaml'
    new_name='changed_abs_mpprocess_input.yaml'
    
    # Define the YAML field to be changed
    abs={'liac':0.011710832742127506,
        'loac':0.012503659055311953,
        'crac':0.01104610137751423}
    # Iterate through each folder
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path) is False:
            continue
        # Find all YAML files in the current folder
        # yaml_files = glob.glob(os.path.join(folder_path, '*.yaml'))
        yaml_files=os.path.join(folder_path,old_name)

        # Iterate through each YAML file

        with open(yaml_files, 'r') as file:
                
                    # Load the YAML content
                    yaml_data = yaml.load(file)
                    
                    for key in abs:
                        yaml_data[key]=abs[key]
                    
                    # Save the updated YAML content
                    with open(os.path.join(folder_path,new_name), 'w') as updated_file:
                        yaml.dump(yaml_data, updated_file)

if __name__ == '__main__':
    #pipeline()
    change_orientation_of_yaml_file()
