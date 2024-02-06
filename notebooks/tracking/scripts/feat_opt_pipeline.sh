#!/bin/bash
base_out_dir="/Users/farzanehwork/Documents/codes/bread/data-clean/generated/"
base_in_dir="/Users/farzanehwork/Documents/codes/bread/data-clean/"
dist=12
edge_fid=0
cell_fid=1
# Run build_cellgraphs
# echo "Running build_cellgraphs for fid=${cell_fid} and out dir is ${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/"
echo ${base_in_dir}
# python -m bread.algo.tracking.build_cellgraphs --segmentations="${base_in_dir}" --out="${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/cellgraphs/" --nn-threshold=${dist} --scale-length=1 --scale-time=1 --cell-fid=${cell_fid} --edge-fid=${edge_fid}

# Run build_assgraphs
# echo "Running build_assgraphs for fid=${cell_fid} and segmentation ${colony_id} and out dir is ${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/"
# python -m bread.algo.tracking.build_assgraphs --cellgraphs "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/cellgraphs/colony001_segmentation/" --out "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/assgraphs" --framediff-min=1 --framediff-max=4
# python -m bread.algo.tracking.build_assgraphs --cellgraphs "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/cellgraphs/colony002_segmentation/" --out "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/assgraphs" --framediff-min=1 --framediff-max=4
# python -m bread.algo.tracking.build_assgraphs --cellgraphs "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/cellgraphs/colony003_segmentation/" --out "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/assgraphs" --framediff-min=1 --framediff-max=4


# Run train.py
# echo "Running training for fid=${cell_fid}"
python ~/Documents/codes/bread/notebooks/tracking/train.py --max-epochs=40 --gamma=0.96 --step-size=512 --assgraphs="${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/assgraphs/" --dataset=colony_0123__dt_1234__t_all --result-dir "${base_out_dir}tracking_cellFid${cell_fid}_edgeFid${edge_fid}_dist${dist}/models/"

# Run compare_external.ipynb
# jupyter nbconvert --execute compare_external.ipynbx
