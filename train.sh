#!/bin/bash
function run_check() {
	status=$?

	if [ $status != 0 ]; then
		echo "run $1-experiment failed"
		exit
	else
		echo "run $1-experiment successfully, come to the next experiment"
	fi 
}
python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1000 \
--weight-fi 0.9 \
--weight-u 0.1 \
--num_classes 7 \
--suffix "acfi_1000"

run_check 1000_acfi_0901
