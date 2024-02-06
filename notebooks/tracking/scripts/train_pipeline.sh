#!/bin/bash
for fid in {0..3}
do  
    # Run build_cellgraphs
    echo "Running build_cellgraphs for dist=${dist} and fid=${fid}"
    for colony_id in {0..5}
    do
        python -m bread.algo.tracking.build_cellgraphs --segmentations "/scratch/izar/labbaf/lineage_tracing/data/data_edited/colony00${colony_id}_segmentation.h5" --out "/scratch/izar/labbaf/lineage_tracing/data/generated/tracking_dist${dist}_Fid${fid}/cellgraphs" --nn-threshold=12 --scale-length=1 --scale-time=1 --fid=${fid}
    done

    # Run build_assgraphs
    echo "Running build_assgraphs for dist=${dist} and fid=${fid}"
    python -m bread.algo.tracking.build_assgraphs --cellgraphs "/scratch/izar/labbaf/lineage_tracing/data/generated/tracking_dist${dist}_Fid${fid}/cellgraphs/*/" --out "/scratch/izar/labbaf/lineage_tracing/data/generated/tracking_dist${dist}_Fid${fid}/assgraphs" --framediff-min=1 --framediff-max=4
    # Run train.py
    echo "Running training for dist=${dist} and fid=${fid}"
    python train.py --max-epochs=40 --gamma=0.96 --step-size=512 --dataset=colony_012345__dt_1234__t_all
done
# Run compare_external.ipynb
# jupyter nbconvert --execute compare_external.ipynb