
import pdb
import numpy as np
import  argparse
import json
import os
import re
import csv
import shutil
if __name__ == '__main__':
  try:
      os.makedirs('image_list_angles')
  except:
      pass
  ori_list=np.linspace(0,180, num=19,dtype=int)
# Overlap_threshold_of_angle_180_yellow_is_the_projection_of_3d_model.png
  final_result=[['orientation','liquor','loop','crystal']]
  for angle in ori_list:
      pth=os.path.join('orientation_{}_save_data'.format(angle),'ResultData','absorption_coefficient')
      shutil.copyfile(os.path.join(pth,'Overlap_threshold_of_angle_{}_yellow_is_the_projection_of_3d_model.png'.format(angle)), './image_list_angles/angle_{}.png'.format(angle))
      with open(os.path.join(pth,'median_coefficients_with_percentage.json')) as f:
          data = json.load(f)
      final_result.append([angle,data[2][2],data[3][2],data[4][2]])
      
      filename='cld_1704_7_3p5kev_abs_orientation.csv'
      with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write each row of data to the CSV file
            for r in final_result:
                writer.writerow(r)