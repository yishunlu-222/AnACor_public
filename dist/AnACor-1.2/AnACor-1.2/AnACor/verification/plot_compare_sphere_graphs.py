import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
import os
import matplotlib.lines as mlines
def sphere_table_angle_zero ( mu , R ) :
    muR = mu * R
    A = 3 / (2 * (muR ** 3)) * (0.5 - np.exp( -2 * muR ) * (0.5 + muR + muR ** 2))
    return A


re = sphere_table_angle_zero( 1 , 1 )
print( re )


def sphere_table_angle_90 ( mu , R ) :
    muR = mu * R
    A = 3 / (4 * muR) * (0.5 - 1 / (16 * (muR) ** 2) * (1 - np.exp( -4 * muR ) * (1 - 4 * muR)))
    return A


re2 = sphere_table_angle_90( 1 , 1 )
print( re2 )
# pdb.set_trace()

def sphere_tables ( table ,second=True ) :
    correct = []
    for i , d in enumerate( table ) :
        if second:
            if i % 2 == 0 :
                correct.append( 1 / d )
        else:
            correct.append( 1 / d )
    return np.array( correct )


def extract_error ( data,index=1 ) :
    errors = []
    for i in data :
        errors.append( i[index] )

    return np.array( errors )





# https://it.iucr.org/Cb/ch6o3v0001/
sphere_tables_p1=np.array( [1.1609, 1.1609, 1.1609, 1.1607, 1.1606, 1.1603, 1.1600, 1.1597, 1.1593, 1.1589, 1.1586, 1.1582, 1.1579, 1.1575, 1.1572, 1.1570, 1.1568, 1.1567, 1.1567])
sphere_tables_p2=np.array( [1.3457, 1.3456, 1.3452, 1.3447, 1.3439, 1.3428, 1.3415, 1.3400, 1.3383, 1.3366, 1.3348, 1.3331, 1.3313, 1.3297, 1.3282, 1.3271, 1.3262, 1.3256, 1.3254])
sphere_tables_p3=np.array([1.5574, 1.5571, 1.5561, 1.5546, 1.5525, 1.5497, 1.5463, 1.5426, 1.5383, 1.5339, 1.5293, 1.5248, 1.5204, 1.5162, 1.5126, 1.5096, 1.5074, 1.5059, 1.5055])
sphere_tables_p5 = [2.0755 , 2.0743 , 2.0706 , 2.0647 , 2.0565 , 2.0462 , 2.0340 , 2.0204 , 2.0056 , 1.9901 , 1.9745 ,
                    1.9592 , 1.9445 , 1.9311 , 1.9194 , 1.9097 , 1.9024 , 1.8979 , 1.8964]
sphere_tables_1 = [4.1237 , 4.1131 , 4.0815 , 4.0304 , 3.9625 , 3.8816 , 3.7917 , 3.6966 , 3.6001 , 3.5048 , 3.4135 ,
                   3.3280 , 3.2499 , 3.1807 , 3.1216 , 3.0738 , 3.0383 , 3.0163 , 3.0090]
sphere_tables_2 = [13.998 , 13.819 , 13.320 , 12.593 , 11.746 , 10.873 , 10.034 , 9.262 , 8.570 , 7.961 , 7.431 ,
                   6.975 , 6.587 , 6.261 , 5.9942 , 5.7842 , 5.6307 , 5.5365 , 5.5041]
pth='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/analytical_absorption_corrrection_verification/sphere/paper'
save_name='sphere'
filenamep1='sphere_sample_1_mur_0.1_0.3.json'
filenamep5='sphere_sample_1_mur_0.5_0.3.json'
filename1='sphere_sample_1_mur_1_0.3_0.01.json'
filenamep1_s='sphere_sample_1_mur_0.1_0.1_0.01.json'
filenamep5_s='sphere_sample_100_mur_0.5_0.1_0.02.json'
filename1_s='sphere_sample_100_mur_1.0_0.1_0.02.json'



filenamep1='sphere_sample_2000_mur_0.1_0.3_0.01.json'
filenamep5='sphere_sample_2000_mur_0.5_0.3_0.01.json'
filename1='sphere_sample_2000_mur_1_0.3_0.01.json'
# filenamep1_s='sphere_sample_2000_mur_0.1_0.1_0.01.json'
# filenamep5_s='sphere_sample_200000_mur_0.5_0.1_0.02.json'
# filename1_s='sphere_sample_2000_mur_1.0_0.1_0.02.json'
filenamep1_s='sphere_sample_54000_mur_0.1_0.1_0.01.json'
filenamep5_s='sphere_sample_54000_mur_0.5_0.1_0.01.json'
filename1_s='sphere_sample_54000_mur_1.0_0.1_0.01.json'
correct_p1=sphere_tables( sphere_tables_p1 )
correct_p2=sphere_tables( sphere_tables_p2 )
correct_p5=sphere_tables( sphere_tables_p5 )
correct_1=sphere_tables( sphere_tables_1 )
correct_2=sphere_tables( sphere_tables_2 )


with open( os.path.join(pth,filenamep1) ) as f1 :
    datap1 = json.load( f1 )
with open( os.path.join(pth,filenamep5) ) as f2 :
    datap5 = json.load( f2 )
with open( os.path.join(pth,filename1) ) as f3 :
    data1 = json.load( f3 )

with open( os.path.join(pth,filenamep1_s) ) as f1 :
    datap1_s = json.load( f1 )
with open( os.path.join(pth,filenamep5_s) ) as f2 :
    datap5_s = json.load( f2 )
with open( os.path.join(pth,filename1_s) ) as f3 :
    data1_s = json.load( f3 )

errorsp1 = extract_error( datap1 )
errorsp5 = extract_error( datap5 )
errors1 = extract_error( data1 )
errorsp1_s = extract_error( datap1_s,index=2  )
errorsp5_s = extract_error( datap5_s,index=2 )
errors1_s = extract_error( data1_s,index=2 )

result_p1 = np.abs( correct_p1 - errorsp1 ) / (errorsp1) * 100 
result_p5 = np.abs( correct_p5 - errorsp5 ) / (errorsp5) * 100
result_1 = np.abs( correct_1 - errors1 ) / (errors1) * 100
result_p1_s = np.abs( correct_p1 - errorsp1_s ) / (errorsp1_s) * 100
result_p5_s = np.abs( correct_p5 - errorsp5_s ) / (errorsp5_s) * 100
result_1_s = np.abs( correct_1 - errors1_s ) / (errors1_s) * 100
pdb.set_trace()
result_p1=result_p1[1:]
result_p5=result_p5[1:]
result_1=result_1[1:]
result_p1_s=result_p1_s[1:]
result_p5_s=result_p5_s[1:]
result_1_s=result_1_s[1:]

xx = list( range(len( result_p1 ) ) )
angle_list = np.linspace( start = 10 , stop = 90 , num = len( result_p1 )  , endpoint = True ).astype(int)
plt.figure(figsize = (20,20))

scatter1 = plt.scatter(angle_list, result_p1, color='#888AC3', edgecolors='black', marker='^', s=800)
scatter2 = plt.scatter(angle_list, result_p5,color='#888AC3', edgecolors='black', marker='o', s=800)
scatter3 = plt.scatter(angle_list, result_1,color='#888AC3', edgecolors='black', marker='s', s=800)
plt.scatter(angle_list, result_p1_s, color='#F99F46', edgecolors='black', marker='^', s=800)
plt.scatter(angle_list, result_p5_s, color='#F99F46', edgecolors='black', marker='o', s=800)
plt.scatter(angle_list, result_1_s, color='#F99F46', edgecolors='black', marker='s', s=800)

legend_triangle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='^', markersize=40, label='Dim A')
legend_circle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='o', markersize=40, label='Dim B')
legend_square = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='s', markersize=40, label='Dim C')

# Add legend to the plot
plt.legend(handles=[legend_triangle, legend_circle, legend_square], fontsize = 48,loc='upper left')


y =np.ones(len(xx))/2
plt.plot( angle_list, y  , '--',linewidth = 10,color='r')
plt.xticks( ticks = angle_list , labels = angle_list , fontsize = 60 )
plt.yticks( fontsize = 60 )
plt.xlabel( 'Diffraction angles' , fontsize = 80 )
plt.ylabel( 'Absolute percentage errors %  ' , fontsize = 80 )
# plt.legend(fontsize = 48,loc='upper right')
# plt.ylim([0, 0.5])
plt.title( 'Spherical shape' , fontsize = 80 )
plt.tight_layout( )
plt.savefig( '{}percentage errors_nonsamnew.png'.format( save_name ) , dpi = 600 )
# plt.show( )

# with open( os.path.join(pth, filenamep5 )) as f3 :
#     datap5 = json.load( f3 )


# errorsp2 = extract_error( datap2 )
# errorsp5 = extract_error( datap5 )
# errorsp1 = 1 / sphere_tables( errorsp1 )
# errorsp2 = 1 / sphere_tables( errorsp2 )
# errorsp5 = 1 / sphere_tables( errorsp5 )

# result_p2 = np.abs( correct_p2 - errorsp2 ) / (errorsp2) * 100
# result_p5 = np.abs( correct_p5 - errorsp5 ) / (errorsp5) * 100


# pdb.set_trace()

# xx = list( range( len( result_1 ) ) )
# angle_list = np.linspace( start = 0 , stop = 90 , num = len(result_1) , endpoint = True )
# label_list = np.linspace( start = 0 , stop = 90 , num = int(len(result_1)/2)+1 , endpoint = True ).astype(int)

# plt.figure(figsize = (20,20))

# plt.plot( angle_list , result_1 , color = '#888AC3' , label = 'Dim A' , linewidth = 4 , marker = '^' , markersize = 20 )
# plt.plot(angle_list , result_2 , label = 'Dim B' , color = '#25B678' , linewidth = 4 , marker = 'o' , markersize = 20 )
# plt.plot( angle_list , result_3 , label = 'Dim C' , color = '#F99F46' , linewidth = 4 , marker = 's' , markersize = 20 )
# y =np.ones(len(angle_list))/2
# plt.plot( angle_list , y  , '--',linewidth = 4,color='r')
# plt.xticks( ticks = label_list  , labels = label_list  , fontsize = 48 )
# plt.xticks(  fontsize = 48 )

# plt.yticks( fontsize = 48 )
# plt.xlabel( 'Diffraction angles' , fontsize = 48 )
# plt.ylabel( 'Absolute percentage errors %  ' , fontsize = 48 )
# plt.legend( fontsize = 48 )
# # plt.ylim([0, 0.5])
# # plt.title( 'Absolute percentage errors' , fontsize = 48 )
# plt.savefig( '{}percentage errors100.png'.format( save_name ) , dpi = 600 )
# plt.show( )
