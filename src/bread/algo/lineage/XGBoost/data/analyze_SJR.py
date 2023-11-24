import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



top_n = 4
num_timepoints = 10
#coeff = np.polyfit(x,y,1)
#print(coeff[0])
############################
# Readout variables
n_closest_at_t0_right                = 0
n_closest_at_t0_false                = 0
n_closest_mean_t_right               = 0
n_closest_mean_t_false               = 0
n_first_m_at_t0_right                = 0
n_first_m_at_t0_false                = 0
n_bud_orientation_right              = 0
n_bud_orientation_false              = 0
distances_right=np.array([],dtype    = np.float64)
distances_false=np.array([],dtype    = np.float64)
ang_to_cand_maj_right                = np.array([],dtype=np.float64)
ang_to_cand_maj_false                = np.array([],dtype=np.float64)
ang_to_bud_maj_right                 = np.array([],dtype=np.float64)
ang_to_bud_maj_false                 = np.array([],dtype=np.float64)
ang_to_bud_maj_right_2               = np.array([],dtype=np.float64)
ang_to_bud_maj_false_2               = np.array([],dtype=np.float64)
candidate_bud_angles_std_right       = np.array([],dtype=np.float64)
candidate_bud_angles_std_false       = np.array([],dtype=np.float64)
candidate_bud_angles_budpt_std_right = np.array([],dtype=np.float64)
candidate_bud_angles_budpt_std_false = np.array([],dtype=np.float64)
candidate_growth_right               = np.array([],dtype=np.float64)
candidate_growth_false               = np.array([],dtype=np.float64)
candidate_growth_budpt_right         = np.array([],dtype=np.float64)
candidate_growth_budpt_false         = np.array([],dtype=np.float64)
candidate_growth_exvec_right         = np.array([],dtype=np.float64)
candidate_growth_exvec_false         = np.array([],dtype=np.float64)
############################



# load
df_raw = pd.read_csv('data_for_sahand.2022-10-25-edited.csv', sep='|')

# add a column that gives unique values to each bud, which are not unique because different colonies may have buds with the same bud_id 
df = df_raw.assign(bud_colony_id=lambda x: x.bud_id + x.colony_id / 10)

# collect all unique bud IDs to loop through below
bud_colony_id_unique = df['bud_colony_id'].unique()

# create output array for training (rows: neighbors (we will choose 4 closest), columns: features)
# fill with -1's
out_full = np.full((top_n,60,bud_colony_id_unique.size), -1, dtype=np.float64)
out_small= np.full((top_n,12,bud_colony_id_unique.size), -1, dtype=np.float64)
out_GT   = np.full((bud_colony_id_unique.size,1),        -1, dtype=np.uint8)
out_bc_ID= np.full((bud_colony_id_unique.size,1),        -1, dtype=np.float64)


for ibud_colony_id, bud_colony_id in enumerate(bud_colony_id_unique):
  # read out table that includes all candidates for a bud
  bud_info                  = df.loc[df['bud_colony_id'] == bud_colony_id]
  # save bud_colony_id
  out_bc_ID[ibud_colony_id] = bud_colony_id
  # read out table that includes all candidates that were around at t = 0, ignore others for the rest of this analysis
  candidates_at_t0          = bud_info.loc[bud_info['time_since_budding'] == 0]
  # sort but turns out Nicole's table is already sorted
  candidates_at_t0_sorted   = candidates_at_t0.sort_values(by=['dist'], ascending=True)
  # pick top n candidates
  candidates_top_n_ids      = candidates_at_t0_sorted.iloc[0:top_n]['candidate_id']


  # analyze top n candidates
  for icandidate_id, candidate_id in enumerate(candidates_top_n_ids):
    candidate_info        = bud_info.loc[bud_info['candidate_id'] == candidate_id]
    # there is a small chance that not sorted by time
    candidate_info_sorted = candidate_info.sort_values(by=['time_since_budding'], ascending=True)
    # save ground truth
    is_bud                = candidate_info_sorted.iloc[0]['is_budding']
    if is_bud:
      out_GT[ibud_colony_id] = icandidate_id
    # may only have data for certain time points after budding
    # will use this information to decide which columns in 'out' matrix to fill
    times_with_data       = candidate_info_sorted['time_since_budding']
    distances             = candidate_info_sorted['dist']                                                                                        ############
    
    # collect all the data
    budcm_to_budpt_l       = np.array(0 * times_with_data, dtype=np.float64)
    budcm_to_candidatecm_l = np.array(0 * times_with_data, dtype=np.float64)
    expansion_vector_l     = np.array(0 * times_with_data, dtype=np.float64)
    position_bud           = np.array(0 * times_with_data, dtype=np.float64)
    orientation_bud        = np.array(0 * times_with_data, dtype=np.float64)
    
    for i_t, t in enumerate(times_with_data):
      candidatemaj                = np.fromstring(candidate_info_sorted.iloc[i_t]['candidatemaj'].strip('[]'),dtype=np.float64,sep=' ')
      budmaj                      = np.fromstring(candidate_info_sorted.iloc[i_t]['budmaj'].strip('[]'),dtype=np.float64,sep=' ')
      budcm_to_budpt              = np.fromstring(candidate_info_sorted.iloc[i_t]['budcm_to_budpt'].strip('[]'),dtype=np.float64,sep=' ')        ############ ?
      budcm_to_candidatecm        = np.fromstring(candidate_info_sorted.iloc[i_t]['budcm_to_candidatecm'].strip('[]'),dtype=np.float64,sep=' ')  ############ ?
      candidatecm_to_budpt        = budcm_to_budpt - budcm_to_candidatecm
      expansion_vector            = np.fromstring(candidate_info_sorted.iloc[i_t]['expansion_vector'].strip('[]'),dtype=np.float64,sep=' ')      ############ ?
      
      budcm_to_candidatecm_l[i_t] = np.linalg.norm(budcm_to_candidatecm)
      budcm_to_budpt_l[i_t]       = np.linalg.norm(budcm_to_budpt)
      expansion_vector_l[i_t]     = np.linalg.norm(expansion_vector)
      
      # position on candidate where bud appears
      # both matter, the position (preferential toward major axis) as well as the change in position (no movement allowed)
      innerproduct                = np.dot(candidatemaj, candidatecm_to_budpt)
      position_bud[i_t]           = np.arccos(np.absolute(innerproduct) / np.linalg.norm(candidatemaj) / np.linalg.norm(candidatecm_to_budpt))   ############
      
      # orientation of bud with respect to candidate
      innerproduct                = np.dot(budmaj, candidatecm_to_budpt)
      orientation_bud[i_t]        = np.arccos(np.absolute(innerproduct) / np.linalg.norm(budmaj) / np.linalg.norm(candidatecm_to_budpt))         ############
      
    # distances
    out_small[icandidate_id,0,ibud_colony_id] = distances.iloc[0]
    out_small[icandidate_id,1,ibud_colony_id] = distances.max()
    offset = 0
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = distances.iloc[i_t]
      
    # growth
    if times_with_data.size >= 4:
      m       = np.polyfit(times_with_data[0:4],budcm_to_candidatecm_l[0:4],1)[0]
      m_budpt = np.polyfit(times_with_data[0:4],budcm_to_budpt_l[0:4],1)[0]
      m_exvec = np.polyfit(times_with_data[0:4],expansion_vector_l[0:4],1)[0]
      out_small[icandidate_id, 2,ibud_colony_id] = m
      out_small[icandidate_id, 3,ibud_colony_id] = m_budpt
      out_small[icandidate_id, 4,ibud_colony_id] = m_exvec
    offset = 10
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = budcm_to_candidatecm_l[i_t]
    offset = 20
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = budcm_to_budpt_l[i_t]
    offset = 30
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = expansion_vector_l[i_t]
      
    # position and movement around mother
    out_small[icandidate_id,5,ibud_colony_id] = position_bud[0]
    out_small[icandidate_id,6,ibud_colony_id] = position_bud.std()
    offset = 40
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = position_bud[i_t]
      
    # orientation of bud
    out_small[icandidate_id,7,ibud_colony_id] = orientation_bud[0]
    if times_with_data.size >= 4:
      m       = np.polyfit(times_with_data[0:4],orientation_bud[0:4],1)[0]
      out_small[icandidate_id, 8,ibud_colony_id] = m
      out_small[icandidate_id, 9,ibud_colony_id] = orientation_bud[3]
    if times_with_data.size >= 8:
      m       = np.polyfit(times_with_data[0:8],orientation_bud[0:8],1)[0]
      out_small[icandidate_id,10,ibud_colony_id] = m
      out_small[icandidate_id,11,ibud_colony_id] = orientation_bud[7]
      if distances.iloc[0] < 5:
#        plt.plot(times_with_data[0:8],orientation_bud[0:8])
#        plt.title(str(is_bud) + ' m: ' +  str(m))
#        plt.show()
        if (is_bud and orientation_bud[0] > orientation_bud[7]) or  ((not is_bud) and orientation_bud[0] < orientation_bud[7]):
          n_bud_orientation_right += 1
        else:
          n_bud_orientation_false += 1
    offset = 50
    for i_t, t in enumerate(times_with_data):
      out_full[icandidate_id,t + offset,ibud_colony_id] = orientation_bud[i_t]
      


  ############################
  # Test closest bud at time 0
  guess_istrue            = candidates_at_t0_sorted.iloc[0]['is_budding']
  if guess_istrue:
    n_closest_at_t0_right += 1
  else:
    n_closest_at_t0_false += 1


    
  ############################
  # How many neighbors should we typically consider?
  # Summary: 4 is enough
  guess_istrue            = np.any(candidates_at_t0_sorted.iloc[0:top_n]['is_budding'])
  if guess_istrue:
    n_first_m_at_t0_right += 1
  else:
    n_first_m_at_t0_false += 1


    
  ############################
  # Test closest bud at all time points
  # Only consider neighbors that were around at time 0
  candidates           = candidates_at_t0['candidate_id'].unique() # .unique turns dataframe to vector
  candidates_distances = np.array(0 * candidates, dtype=np.float64)
  
  for icandidate_id, candidate_id in enumerate(candidates):
    # look at one specific neighbor
    candidate_info                      = bud_info.loc[bud_info['candidate_id'] == candidate_id]
    # calculate the mean distance to the neighbor over all time points
    candidates_distances[icandidate_id] = candidate_info['dist'].max()
  # find neighbor with smallest average distance
  on_average_closest_candidate = candidates[np.argmin(candidates_distances)]
  guess_istrue                 = candidates_at_t0.loc[candidates_at_t0['candidate_id'] == on_average_closest_candidate].iloc[0]['is_budding']
  if guess_istrue:
    n_closest_mean_t_right += 1
  else:
    n_closest_mean_t_false += 1


    
  ############################
  # Collect statistics on candidates

  for icandidate_id, candidate_id in enumerate(candidates):
    # look at one specific neighbor
    candidate_info = bud_info.loc[bud_info['candidate_id'] == candidate_id]
    is_bud = candidate_info.iloc[0]['is_budding']
    
    candidate_at_t0 = candidate_info.loc[candidate_info['time_since_budding'] == 0]
    


    ############################
    # statistics on distance
    if is_bud:
      # compute distance at time 0
      distances_right = np.append(distances_right, candidate_at_t0['dist'])
    else:
      distances_false = np.append(distances_false, candidate_at_t0['dist'])

    # do not waste time on bad neighbors
    if candidate_at_t0.iloc[0]['dist'] < 5:
      
      # can look at any single time point after budding
      candidate_at_t = candidate_info.loc[candidate_info['time_since_budding'] == 7]
      if not candidate_at_t.empty:
        


        ############################
        # statistics on candidate major/minor axes at specific time point
        # summary: got good results at time 0
        candidatemaj         = np.fromstring(candidate_at_t.iloc[0]['candidatemaj'].strip('[]'),dtype=np.float64,sep=' ')
        candidatecm_to_budpt = np.fromstring(candidate_at_t.iloc[0]['budcm_to_budpt'].strip('[]'),dtype=np.float64,sep=' ') \
                             - np.fromstring(candidate_at_t.iloc[0]['budcm_to_candidatecm'].strip('[]'),dtype=np.float64,sep=' ')
        innerproduct         = np.dot(candidatemaj, candidatecm_to_budpt)
        angle                = np.arccos(np.absolute(innerproduct) / np.linalg.norm(candidatemaj) / np.linalg.norm(candidatecm_to_budpt))
        if is_bud:
          ang_to_cand_maj_right = np.append(ang_to_cand_maj_right, angle)
        else:
          ang_to_cand_maj_false = np.append(ang_to_cand_maj_false, angle)
    


        ############################
        # statistics on bud major/minor axes at specific time point
        # summary: got good changes in bud major-minor axis for different time points 0 versus 6 or 7
        budmaj       = np.fromstring(candidate_at_t.iloc[0]['budmaj'].strip('[]'),dtype=np.float64,sep=' ')
        innerproduct = np.dot(budmaj, candidatecm_to_budpt)
        angle        = np.arccos(np.absolute(innerproduct) / np.linalg.norm(budmaj) / np.linalg.norm(candidatecm_to_budpt))
        if is_bud:
          ang_to_bud_maj_right = np.append(ang_to_bud_maj_right, angle)
        else:
          ang_to_bud_maj_false = np.append(ang_to_bud_maj_false, angle)
          


        ############################
        # statistics on bud major/minor axes at specific time point, but using budcm_to_budpt here
        # summary: 
        innerproduct = np.dot(budmaj, budcm_to_budpt)
        angle        = np.arccos(np.absolute(innerproduct) / np.linalg.norm(budmaj) / np.linalg.norm(budcm_to_budpt))
        if is_bud:
          ang_to_bud_maj_right_2 = np.append(ang_to_bud_maj_right_2, angle)
        else:
          ang_to_bud_maj_false_2 = np.append(ang_to_bud_maj_false_2, angle)
          


      ############################
      # statistics on angular velocity
      # summary: budcm or budpt does not seem to make a big difference
      all_budcm_to_candidatecm   = candidate_info['budcm_to_candidatecm']
      all_candidatemaj           = candidate_info['candidatemaj']
      all_budcm_to_budpt         = candidate_info['budcm_to_budpt']
      all_budcm_to_candidatecm   = candidate_info['budcm_to_candidatecm']
      candidate_bud_angles       = np.array([],dtype=np.float64)
      candidate_bud_angles_budpt = np.array([],dtype=np.float64)
      for ivec, str_budcm_to_candidatecm in enumerate(all_budcm_to_candidatecm):
        vec_budcm_to_candidatecm   = np.fromstring(str_budcm_to_candidatecm.strip('[]'),dtype=np.float64,sep=' ')
        vec_candidatemaj           = np.fromstring(all_candidatemaj.iloc[ivec].strip('[]'),dtype=np.float64,sep=' ')
        innerproduct               = np.dot(vec_budcm_to_candidatecm, vec_candidatemaj)
        angle                      = np.arccos(np.absolute(innerproduct) / np.linalg.norm(vec_budcm_to_candidatecm) / np.linalg.norm(vec_candidatemaj))
        candidate_bud_angles       = np.append(candidate_bud_angles, angle)
        
        vec_budcm_to_budpt         = np.fromstring(all_budcm_to_budpt.iloc[ivec].strip('[]'),dtype=np.float64,sep=' ')
        vec_budcm_to_candidatecm   = np.fromstring(all_budcm_to_candidatecm.iloc[ivec].strip('[]'),dtype=np.float64,sep=' ')
        vec_candidatecm_to_budpt   = vec_budcm_to_budpt - vec_budcm_to_candidatecm
        innerproduct               = np.dot(vec_candidatecm_to_budpt, vec_candidatemaj)
        angle                      = np.arccos(np.absolute(innerproduct) / np.linalg.norm(vec_candidatecm_to_budpt) / np.linalg.norm(vec_candidatemaj))
        candidate_bud_angles_budpt = np.append(candidate_bud_angles, angle)
      if is_bud:
        candidate_bud_angles_std_right       = np.append(candidate_bud_angles_std_right, candidate_bud_angles.std())
        candidate_bud_angles_budpt_std_right = np.append(candidate_bud_angles_budpt_std_right, candidate_bud_angles_budpt.std())
      else:
        candidate_bud_angles_std_false       = np.append(candidate_bud_angles_std_false, candidate_bud_angles.std())
        candidate_bud_angles_budpt_std_false = np.append(candidate_bud_angles_budpt_std_false, candidate_bud_angles_budpt.std())
      
    
  
      ############################
      # statistics on growth
      # summary: probably useless but there could be minor differences
#      all_budcm_to_candidatecm   = candidate_info['budcm_to_candidatecm']
#      all_budcm_to_budpt         = candidate_info['budcm_to_budpt']
#      all_budcm_to_candidatecm   = candidate_info['budcm_to_candidatecm']
      all_time_since_budding     = candidate_info['time_since_budding']
      all_expansion_vector       = candidate_info['expansion_vector']
      candidate_growth           = np.array([],dtype=np.float64)
      candidate_growth_budpt     = np.array([],dtype=np.float64)
      candidate_growth_times     = np.array([],dtype=np.float64)
      candidate_expansion_vectors= np.array([],dtype=np.float64)
      for ivec, str_budcm_to_candidatecm in enumerate(all_budcm_to_candidatecm):
        vec_budcm_to_candidatecm   = np.fromstring(str_budcm_to_candidatecm.strip('[]'),dtype=np.float64,sep=' ')
        vec_budcm_to_budpt         = np.fromstring(all_budcm_to_budpt.iloc[ivec].strip('[]'),dtype=np.float64,sep=' ')
        time_since_budding         = all_time_since_budding.iloc[ivec]
        expansion_vector           = np.fromstring(all_expansion_vector.iloc[ivec].strip('[]'),dtype=np.float64,sep=' ')
        candidate_growth           = np.append(candidate_growth,            np.linalg.norm(vec_budcm_to_candidatecm))
        candidate_growth_budpt     = np.append(candidate_growth_budpt,      np.linalg.norm(vec_budcm_to_budpt))
        candidate_growth_times     = np.append(candidate_growth_times,      time_since_budding)
        candidate_expansion_vectors= np.append(candidate_expansion_vectors, np.linalg.norm(expansion_vector))
#      print(candidate_growth_times)
#      print(candidate_growth)
#      print(candidate_growth_times.size)
      if candidate_growth_times.size >= 4:
        m       = np.polyfit(candidate_growth_times[0:3],candidate_growth[0:3],1)[0]
        m_budpt = np.polyfit(candidate_growth_times[0:3],candidate_growth_budpt[0:3],1)[0]
        m_exvec = np.polyfit(candidate_growth_times[0:3],candidate_expansion_vectors[0:3],1)[0]
        if is_bud:
          candidate_growth_right       = np.append(candidate_growth_right, m)
          candidate_growth_budpt_right = np.append(candidate_growth_budpt_right, m_budpt)
          candidate_growth_exvec_right = np.append(candidate_growth_exvec_right, m_exvec)
#          plt.plot(candidate_growth_times, candidate_growth_budpt)
#          plt.title('Right ' + str(m_budpt))
#          plt.show()
        else:
          candidate_growth_false       = np.append(candidate_growth_false, m)
          candidate_growth_budpt_false = np.append(candidate_growth_budpt_false, m_budpt)
          candidate_growth_exvec_false = np.append(candidate_growth_exvec_false, m_exvec)
#          plt.plot(candidate_growth_times, candidate_growth_budpt)
#          plt.title('False ' + str(m_budpt))
#          plt.show()
      
    
  
#  if ibud_colony_id == 1:
#    break



np.save('out_full.npy',out_full)
np.save('out_small.npy',out_small)
np.save('out_GT.npy',out_GT)
np.save('out_bc_ID.npy',out_bc_ID)




print("Statistics if choose closest neighbor at time 0 only:")
print(n_closest_at_t0_right/(n_closest_at_t0_right + n_closest_at_t0_false))
print(n_closest_at_t0_false/(n_closest_at_t0_right + n_closest_at_t0_false))
print("Statistics if choose closest neighbor for all time points:")
print(n_closest_mean_t_right/(n_closest_mean_t_right + n_closest_mean_t_false))
print(n_closest_mean_t_false/(n_closest_mean_t_right + n_closest_mean_t_false))
print("Statistics if choose closest m neighbors at time 0 only:")
print(n_first_m_at_t0_right/(n_first_m_at_t0_right + n_first_m_at_t0_false))
print(n_first_m_at_t0_false/(n_first_m_at_t0_right + n_first_m_at_t0_false))
print("Statistics for bud orientation")
print(n_bud_orientation_right/(n_bud_orientation_right + n_bud_orientation_false))
print(n_bud_orientation_false/(n_bud_orientation_right + n_bud_orientation_false))
#print("Distances if is bud:")
#print(distances_right)
#print("Distances if is not bud:")
#print(distances_false)
#plt.hist(distances_right, bins=20)
#plt.show()
#print(np.sort(distances_right)[::-1])
#print(np.argsort(distances_right)[::-1])

plt.subplot(8,2, 1)
plt.hist(ang_to_cand_maj_right, bins=20, facecolor='g')
plt.subplot(8,2, 2)
plt.hist(ang_to_cand_maj_false, bins=20, facecolor='g')
plt.subplot(8,2, 3)
plt.hist(ang_to_bud_maj_right, bins=20)
plt.subplot(8,2, 4)
plt.hist(ang_to_bud_maj_false, bins=20)
plt.subplot(8,2, 5)
plt.hist(ang_to_bud_maj_right_2, bins=20)
plt.subplot(8,2, 6)
plt.hist(ang_to_bud_maj_false_2, bins=20)
plt.subplot(8,2, 7)
plt.hist(candidate_bud_angles_std_right, bins=20, facecolor='r', range=[0, 0.7])
plt.subplot(8,2, 8)
plt.hist(candidate_bud_angles_std_false, bins=20, facecolor='r', range=[0, 0.7])
plt.subplot(8,2, 9)
plt.hist(candidate_bud_angles_budpt_std_right, bins=20, facecolor='k', range=[0, 0.7])
plt.subplot(8,2,10)
plt.hist(candidate_bud_angles_budpt_std_false, bins=20, facecolor='k', range=[0, 0.7])
plt.subplot(8,2,11)
plt.hist(candidate_growth_right, bins=20, range=[0, 5])
plt.subplot(8,2,12)
plt.hist(candidate_growth_false, bins=20, range=[0, 5])
plt.subplot(8,2,13)
plt.hist(candidate_growth_budpt_right, bins=20, range=[0, 5])
plt.subplot(8,2,14)
plt.hist(candidate_growth_budpt_false, bins=20, range=[0, 5])
plt.subplot(8,2,15)
plt.hist(candidate_growth_exvec_right, bins=20, range=[0, 5])
plt.subplot(8,2,16)
plt.hist(candidate_growth_exvec_false, bins=20, range=[0, 5])
plt.show()

