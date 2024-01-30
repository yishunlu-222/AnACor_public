import os
import glob
import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True 
import pdb
# Define the path to the root directory containing the folders
def create_merge_bash():
    # the bash files dials_anacor_several_datasets.sh for the firstxx datasets
    root_directory = '/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/multiple_datasets/'
    single_directory='/dls/i23/data/2023/nr29467-16/processing/ramona/anacor/cld_1704_7_3p5kev/anom_false/single_datasets/'
    experiment_format= '/dls/i23/data/2023/nr29467-16/processed/Cld/for_anacor/cld_1704_7/3500ev_1/xia2-dials-P1/DataFiles/nr29467v16_x3500ev1_SAD_SWEEP1.expt'
    term1='3500ev_1'
    term2='x3500ev1'
    dataset='auto'
    dataset_list=['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                  '16','17','18','19','20','21','22','23','24','25','26']
    anomalous = 'False'

    for folder in dataset_list:
        refl_list=[]
        expt_list=[]
        name = 'first_'+folder
        pth = os.path.join(root_directory,name)
        try:

            os.makedirs(pth)
            print('Folder {}is  created'.format(folder))
        except:
            print('Folder {} already exists'.format(folder))

        for i in range(int(folder)):
            index = i+1
            refl_name = os.path.join(single_directory,str(index),str(index)+'_save_data','ResultData','dials_output','anacor_'+str(index)+'.refl')
            refl_list.append(refl_name)
            new_term_1 = term1[:-1]+str(index)
            new_term_2 = term2[:-1]+str(index)
            expt_name = experiment_format.replace(term1,new_term_1).replace(term2,new_term_2)
            expt_list.append(expt_name)

        refl_str = ', '.join(refl_list)
        expt_str = ', '.join(expt_list)
        with open( os.path.join( pth , "dials_anacor_several_datasets.sh" ) , "w" ) as f :
            f.write( "#!/bin/sh\n" )
            f.write( "{}\n".format( 'source /dls/science/groups/i23/yishun/dials_yishun/dials' ) )
            f.write( "dials.scale  {0} {1} "
                "anomalous={3}  physical.absorption_correction=False physical.analytical_correction=True "
                "output.reflections=result_{2}_ac.refl  output.html=result_{2}_ac.html "
                "output{{log={2}_ac_log.log}} output{{unmerged_mtz={2}_unmerged_ac.mtz}} output{{merged_mtz={2}_merged_ac.mtz}} "
                "\n".format( refl_str , expt_str , dataset,anomalous ).replace( ',' , ' ' ) )
            f.write( "\n" )
            f.write( "dials.scale  {0} {1}  "
                        "anomalous={3}  physical.absorption_level=high physical.analytical_correction=True "
                        "output.reflections=result_{2}_acsh.refl  output.html=result_{2}_acsh.html "
                        "output{{log={2}_acsh_log.log}}  output{{unmerged_mtz={2}_unmerged_acsh.mtz}} "
                        "output{{merged_mtz={2}_merged_acsh.mtz}} "
                        "\n".format( refl_str , expt_str , dataset,anomalous ).replace( ',' , ' ' )  )
            f.write( "{} \n".format( 'module load ccp4' ) )
            f.write( "mtz2sca {}_merged_acsh.mtz   \n".format(dataset) )
            f.write( "mtz2sca {}_merged_ac.mtz   \n".format(dataset) )

        print('bash file of {} is created'.format(name))

if __name__ == '__main__':
    create_merge_bash()
    #change_contents_of_yaml_file()
