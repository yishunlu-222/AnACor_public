import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
import os
import matplotlib.lines as mlines


def cylinder_tables ( table ,second=True ) :
    correct = []
    for i , d in enumerate( table ) :
        if second:
            if i % 2 == 0 :
                correct.append( 1 / d )
        else:
            correct.append( 1 / d )
    return np.array( correct )


def extract_error ( data ) :
    errors = []
    for i in data :
        errors.append( i[1] )

    return np.array( errors )


#https://it.iucr.org/Cb/ch6o3v0001/
cylinder_tables_p1=np.array([1.1843,	1.1843,	1.1842,	1.1840,	1.1838,	1.1835,	1.1832,	1.1828,	1.1823,	1.1818,	1.1813,	1.1808,	1.1802,	1.1798	,1.1793	,1.1790	,1.1787,	1.1785,	1.1785])
cylinder_tables_p2=np.array([1.4009, 1.4007, 1.4002, 1.3995, 1.3984, 1.3970, 1.3953, 1.3934, 1.3912, 1.3889, 1.3865, 1.3841, 1.3818, 1.3796, 1.3777, 1.3761, 1.3749, 1.3741, 1.3739])
cylinder_tables_p3=np.array( [1.6548, 1.6544, 1.6531, 1.6510, 1.6481, 1.6443, 1.6398, 1.6347, 1.6290, 1.6230, 1.6169, 1.6108, 1.6049, 1.5994, 1.5946, 1.5906, 1.5876, 1.5857, 1.5851])
cylinder_tables_p5 = [2.2996 , 2.2979 , 2.2926 , 2.2840 , 2.2721 , 2.2572 , 2.2398 , 2.2204 , 2.1996 , 2.1781 , 2.1564 ,
                      2.1352 , 2.1152 , 2.0969 , 2.0809 , 2.0677 , 2.0579 , 2.0518 , 2.0497]
cylinder_tables_1 = [5.0907 , 5.0724 , 5.0185 , 4.9323 , 4.8196 , 4.6877 , 4.5439 , 4.3948 , 4.2461 , 4.1022 , 3.9664 ,
                     3.8413 , 3.7286 , 3.6298 , 3.5462 , 3.4790 , 3.4295 , 3.3990 , 3.3886]
cylinder_tables_2 = [21.43 , 21.00 , 19.84 , 18.24 , 16.50 , 14.824 , 13.311 , 11.995 , 10.871 , 9.921 , 9.122 , 8.452 ,
                     7.895 , 7.435 , 7.063 , 6.773 , 6.562 , 6.433 , 6.389]

correct_p1 = cylinder_tables( cylinder_tables_p1 )
correct_p2 = cylinder_tables( cylinder_tables_p2 )
correct_p3 = cylinder_tables( cylinder_tables_p3 )
correct_p5 = cylinder_tables( cylinder_tables_p5 )

correct_1 = cylinder_tables( cylinder_tables_1 )
correct_2 = cylinder_tables( cylinder_tables_2 )

# correct_2[0] = cylinder_table_angle_zero ( 1 , 2 )
# correct_1[0] = cylinder_table_angle_zero ( 1 , 1 )
# correct_p5[0] = cylinder_table_angle_zero ( 1 , 0.5 )

# correct_2[-1] = cylinder_table_angle_90 ( 1 , 2 )
# correct_1[-1] = cylinder_table_angle_90 ( 1 , 1 )
# correct_p5[-1] = cylinder_table_angle_90 ( 1 , 0.5 )
print( correct_p1 )
print( correct_p2 )
print( correct_p5 )
print( correct_1 )
print( correct_2 )

pth='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/analytical_absorption_corrrection_verification/cylinder'
save_name='cylinder'



filenamep1='cylinder_sample_1_mur_0.1_0.3.json'
filenamep2='cylinder_sample_1_mur_0.2_0.3.json'
filenamep3='cylinder_sample_1_mur_0.3_0.3.json'
filenamep5='cylinder_sample_1_mur_0.5_0.3.json'
filename1='cylinder_sample_1_mur_1_0.3.json'
filename2='cylinder_sample_1_mur_2_0.3.json'

filenamep1='cylinder_sample_1_mur_0.1_0.1_l_1_mu_0.1.json'
filenamep5='cylinder_sample_1_mur_0.5_0.1_l_1_mu_0.1.json'
filename1='cylinder_sample_1_mur_1_0.1_l_1_mu_0.1.json'


filenamep1_s='cylinder_sample_2000_mur_0.1_0.3_l_50.json'
filenamep2_s='cylinder_sample_2000_mur_0.2_0.3.json'
filenamep3_s='cylinder_sample_2000_mur_0.3_0.3.json'
filenamep5_s='cylinder_sample_2000_mur_0.5_0.3_l_50.json'
filename1_s='cylinder_sample_2000_mur_1_0.3_l_50.json'
filename2_s='cylinder_sample_2000_mur_2_0.3.json'
filenamep1_s='cylinder_sample_2000_mur_0.1_0.1_l_50_mu_0.01.json'
filenamep5_s='cylinder_sample_2000_mur_0.5_0.1_l_50_mu_0.1.json'
filename1_S='cylinder_sample_2000_mur_1_0.1_l_50_mu_0.1.json'

with open( os.path.join(pth,filenamep1) ) as f1 :
    datap1 = json.load( f1 )
with open( os.path.join(pth, filenamep5 )) as f3 :
    datap5 = json.load( f3 )
with open( os.path.join(pth, filename1 )) as f4 :
    data1 = json.load( f4 )
with open( os.path.join(pth,filenamep1_s) ) as f1 :
    datap1_s = json.load( f1 )
with open( os.path.join(pth, filenamep5_s )) as f3 :
    datap5_s = json.load( f3 )
with open( os.path.join(pth, filename1_s )) as f4 :
    data1_s = json.load( f4 )
# array([0.26445344, 0.27490508, 0.26727876, 0.27113762, 0.27444762,
#        0.26720538, 0.32270223, 0.3190568 , 0.27831181, 0.26929138])

errorsp1 = 1 / cylinder_tables( extract_error( datap1 ),second=False)
errorsp5 = 1 / cylinder_tables(extract_error( datap5 ),second=False)
error_1 = 1 / cylinder_tables(extract_error( data1 ),second=False)


result_p1 = np.abs( correct_p1 - errorsp1 ) / (errorsp1) * 100 
result_p5 = np.abs( correct_p5 - errorsp5 ) / (errorsp5) * 100
result_1 = np.abs( correct_1 - error_1 ) / (error_1) * 100


errorsp1_s = 1 / cylinder_tables( extract_error( datap1_s ),second=False)
errorsp5_s = 1 / cylinder_tables(extract_error( datap5_s ),second=False)
error_1_s = 1 / cylinder_tables(extract_error( data1_s ),second=False)

result_p1_s = np.abs( correct_p1 - errorsp1_s ) / (errorsp1_s) * 100
result_p5_s = np.abs( correct_p5 - errorsp5_s ) / (errorsp5_s) * 100
result_1_s = np.abs( correct_1 - error_1_s ) / (error_1_s) * 100

result_p1=result_p1[1:]
result_p5=result_p5[1:]
result_1=result_1[1:]
result_p1_s=result_p1_s[1:]
result_p5_s=result_p5_s[1:]
result_1_s=result_1_s[1:]
pdb.set_trace( )
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
plt.legend(handles=[legend_triangle, legend_circle, legend_square], fontsize = 48,loc='upper right')


y =np.ones(len(xx))/2
plt.plot( angle_list, y  , '--',linewidth = 10,color='r')
plt.xticks( ticks = angle_list , labels = angle_list , fontsize = 60 )
plt.yticks( fontsize = 60 )
plt.xlabel( 'Diffraction angles' , fontsize = 80 )
plt.ylabel( 'Absolute percentage errors %  ' , fontsize = 80 )
# plt.legend(fontsize = 48,loc='upper right')
# plt.ylim([0, 0.5])
# plt.title( 'Absolute percentage errors' , fontsize = 48 )
plt.tight_layout( )
plt.savefig( '{}percentage errors1000.png'.format( save_name ) , dpi = 600 )
plt.show( )




# result_p1_s=result_p1_s[1:]
# result_p2_s=result_p2_s[1:]
# result_p3_s=result_p3_s[1:]
# result_p5=result_p5[1:]
# result_1=result_1[1:]
# result_2=result_2[1:]
# xx = list( range( len( result_1 ) ) )
# angle_list = np.linspace( start = 10 , stop = 90 , num = len(result_1) , endpoint = True )
# label_list = np.linspace( start = 10 , stop = 90 , num = int(len(result_1)/2)+1 , endpoint = True ).astype(int)

# plt.figure(figsize = (20,20))

# # plt.plot( angle_list , result_1 , color = '#888AC3' , label = 'Dim A' , linewidth = 4 , marker = '^' , markersize = 20 )
# # plt.plot(angle_list , result_2 , label = 'Dim B' , color = '#25B678' , linewidth = 4 , marker = 'o' , markersize = 20 )
# # plt.plot( angle_list , result_3 , label = 'Dim C' , color = '#F99F46' , linewidth = 4 , marker = 's' , markersize = 20 )



# line1 = plt.plot(angle_list, result_p5, color='#888AC3', label = 'Dim A' , marker='^', markersize=50, linewidth = 10, linestyle='-')
# line2 = plt.plot(angle_list, result_1, color='#25B678', label = 'Dim B' , marker='o', markersize=50, linewidth = 10, linestyle='-')
# line3 = plt.plot(angle_list, result_2, color='#F99F46', label = 'Dim C' , marker='s', markersize=50, linewidth = 10, linestyle='-')
# # plt.scatter(xx, errors4, color='#F99F46', edgecolors='black', marker='^', s=800)
# # plt.scatter(xx, errors5, color='#F99F46', edgecolors='black', marker='o', s=800)
# # plt.scatter(xx, errors6, color='#F99F46', edgecolors='black', marker='s', s=800)

# # legend_triangle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='^', markersize=40, label='Dim A')
# # legend_circle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='o', markersize=40, label='Dim B')
# # legend_square = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='s', markersize=40, label='Dim C')

# # Add legend to the plot
# # plt.legend(handles=[legend_triangle, legend_circle, legend_square], fontsize = 48,loc='upper right')
# plt.legend( fontsize = 60 ,loc='upper right')


# y =np.ones(len(xx))/2
# plt.plot( angle_list, y  , '--',linewidth = 10,color='r')
# plt.xticks( ticks = label_list , labels = label_list , fontsize = 60 )
# plt.yticks( fontsize = 60 )
# plt.xlabel( 'Diffraction angles' , fontsize = 80 )
# plt.ylabel( 'Absolute percentage errors %  ' , fontsize = 80 )
# # plt.legend(fontsize = 48,loc='upper right')
# # plt.ylim([0, 0.5])
# # plt.title( 'Absolute percentage errors' , fontsize = 48 )
# plt.tight_layout( )
# plt.savefig( '{}percentage errors1000.png'.format( save_name ) , dpi = 600 )
# plt.show( )
