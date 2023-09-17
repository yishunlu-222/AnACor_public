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


def sphere_tables ( table ,second=True ) :
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
        if i[0]==0:
            continue
        errors.append( i[1] )

    return np.array( errors )


save_name = 'rectangular'
filename1 = './rectangular sample 1 w1_h1.json'
filename2 = './rectangular sample 1 w1_h2.json'
filename3 = './rectangular sample 1 w2_h1.json'
pth='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/analytical_absorption_corrrection_verification/rect'
save_name='rect'
filename1 = 'rect_sample_w_1_h_3_0.03.json'
filename2 = 'rect_sample_w_3_h_1_0.03.json'
filename3 = 'rect_sample_w_3_h_3_0.03.json'

filename1='rect_sample_w_100_h_50_0.3.json'
filename2='rect_sample_w_100_h_100_0.3.json'
filename3='rect_sample_w_100_h_150_0.3.json'
filename4='rect_sample_2000_w_100_h_50_l_100_0.3.json'
filename5='rect_sample_2000_w_100_h_100_l_111_0.3.json'
filename6='rect_sample_2000_w_100_h_150_l_100_0.3.json'
with open( os.path.join(pth,filename1) ) as f1 :
    data1 = json.load( f1 )
with open(  os.path.join(pth,filename2) ) as f2 :
    data2 = json.load( f2 )
with open( os.path.join(pth, filename3 )) as f3 :
    data3 = json.load( f3 )
with open( os.path.join(pth, filename4 )) as f4 :
    data4 = json.load( f4 )
with open( os.path.join(pth, filename5 )) as f5 :
    data5 = json.load( f5 )
with open( os.path.join(pth, filename6 )) as f6 :
    data6 = json.load( f6 )

errors1 = extract_error( data1 )
errors2 = extract_error( data2 )
errors3 = extract_error( data3 )
errors4 = extract_error( data4 )
errors5 = extract_error( data5 )
errors6 = extract_error( data6 )




xx = list( range( len( errors1 ) ) )
angle_list = np.linspace( start = 10 , stop = 90 , num = len(errors1) , endpoint = True )
label_list = np.linspace( start = 10 , stop = 90 , num = int(len(errors1)/2)+1 , endpoint = True ).astype(int)

plt.figure(figsize = (20,20))

# plt.plot( angle_list , result_1 , color = '#888AC3' , label = 'Dim A' , linewidth = 4 , marker = '^' , markersize = 20 )
# plt.plot(angle_list , result_2 , label = 'Dim B' , color = '#25B678' , linewidth = 4 , marker = 'o' , markersize = 20 )
# plt.plot( angle_list , result_3 , label = 'Dim C' , color = '#F99F46' , linewidth = 4 , marker = 's' , markersize = 20 )



# line1 = plt.plot(angle_list, errors1, color='#888AC3', label = 'Dim A' , marker='^', markersize=50, linewidth = 10, linestyle='-')
# line2 = plt.plot(angle_list, errors2, color='#25B678', label = 'Dim B' , marker='o', markersize=50, linewidth = 10, linestyle='-')
# line3 = plt.plot(angle_list, errors3, color='#F99F46', label = 'Dim C' , marker='s', markersize=50, linewidth = 10, linestyle='-')
# plt.legend( fontsize = 60 ,loc='upper right')
plt.scatter(angle_list, errors1, color='#888AC3', edgecolors='black', marker='^', s=800)
plt.scatter(angle_list, errors2, color='#888AC3', edgecolors='black', marker='o', s=800)
plt.scatter(angle_list, errors3, color='#888AC3', edgecolors='black', marker='s', s=800)
plt.scatter(angle_list, errors4, color='#F99F46', edgecolors='black', marker='^', s=800)
plt.scatter(angle_list, errors5, color='#F99F46', edgecolors='black', marker='o', s=800)
plt.scatter(angle_list, errors6, color='#F99F46', edgecolors='black', marker='s', s=800)

legend_triangle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='^', markersize=40, label='Dim A')
legend_circle = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='o', markersize=40, label='Dim B')
legend_square = mlines.Line2D([], [], color='none', markeredgecolor='black', marker='s', markersize=40, label='Dim C')
plt.legend(handles=[legend_triangle, legend_circle, legend_square], fontsize = 48,loc='upper right')
# Add legend to the plot




y =np.ones(len(xx))/2
plt.plot( angle_list, y  , '--',linewidth = 10,color='r')
plt.xticks( ticks = label_list , labels = label_list , fontsize = 60 )
plt.yticks( fontsize = 60 )
plt.xlabel( 'Diffraction angles' , fontsize = 80 )
plt.ylabel( 'Absolute percentage errors %  ' , fontsize = 80 )
# plt.legend(fontsize = 48,loc='upper right')
# plt.ylim([0, 0.5])
# plt.title( 'Absolute percentage errors' , fontsize = 48 )
plt.tight_layout( )
plt.savefig( '{}percentage errors1000.png'.format( save_name ) , dpi = 600 )
plt.show( )
