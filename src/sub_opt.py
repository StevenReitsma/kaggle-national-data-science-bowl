__author__ = 'Robbert & Poll'
import pandas
from pandas import DataFrame
import numpy as np
import scipy.stats as st

header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,\
appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,\
artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,\
copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,\
copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,\
copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,\
copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,\
ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,\
diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,\
echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,\
echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,\
euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,\
fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,\
hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,\
hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,\
hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,\
hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,\
hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,\
jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,\
pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,\
shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,\
siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,\
siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,\
siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,\
siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,\
trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,\
tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,\
unknown_unclassified".split(',')

_threshold = 0.9

def load_file(path='../data/submission1/out.csv'):
         probabilities = pandas.io.parsers.read_csv(path,sep=',')
         return probabilities.as_matrix(),probabilities

def add_probs(newfile,index,probs):
    probs[:] = 0
    probs[index] = 1
    newfile.append(probs)
    return newfile

def combine_probs(prob1, prob2):
    counter = 0
    newfile = []
    for prob in prob1:
        if (np.max(prob) > _threshold and np.max(prob2[counter]) < _threshold):
            index = np.argmax(prob)
            newfile = add_probs(newfile,index,prob)
#            print (str(index) + " from 1st submission, picture " + str(counter))
        elif (np.max(prob) < _threshold and np.max(prob2[counter]) > _threshold):
            index = np.argmax(prob2[counter])
            newfile = add_probs(newfile,index,prob2[counter])
#            print (str(index) + " from 2nd submission, picture " + str(counter))
        elif (np.max(prob) > _threshold and np.max(prob2[counter]) > _threshold):
            index1 = np.argmax(prob)
            index2 = np.argmax(prob2[counter])
            if index1 == index2:
#                print (str(index1) + " from both submissions, picture " + str(counter))
                newfile = add_probs(newfile,index1,prob)
#            else:
#                print("Different indices! picture " + str(counter))
            else:
                newline = np.mean(np.vstack([prob,prob2[counter]]),axis=0)
                newfile.append(newline)

        else:
            newline = np.mean(np.vstack([prob,prob2[counter]]),axis=0)
            newfile.append(newline)
        counter+=1
#    print newfile
    return newfile

def entropy(probs):
    entropy = 0
    for prob in probs:
        entropy += -prob*np.log(prob)
    return entropy

def entropy_compare(probs1,probs2):
    counter = 0
    newfile = []
    for prob in probs1:
        if  entropy(prob) < entropy(probs2[counter]):
            newfile.append(prob)
#            print (str(index) + " from 1st submission, picture " + str(counter))
        else:
            newfile.append(probs2[counter])
#            print (str(index) + " from 2nd submission, picture " + str(counter))
        counter+=1
    return newfile

if __name__ == '__main__':
    probs1,df1 = load_file()
    probs2,df2 = load_file(path='../data/submission2/out.csv')
    labels = probs1[:,0]
    probs1 = np.delete(probs1,0,1)
    probs2 = np.delete(probs2,0,1)
    newfile = entropy_compare(probs1,probs2)
    df = DataFrame(newfile,index=labels,columns=header)
    df.index.name = 'image'
#    df.set_index(labels)
#    print newfile
#    df1.replace(df1.index,newfile)
    df.to_csv('out.csv')