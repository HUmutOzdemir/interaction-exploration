echo "RL Training for RGB Agent" 
bash interaction_exploration/tools/train_comparison.sh
echo "RL Training is Completed" 

echo "Affordance Dataset Creation is Started" 
bash affordance_seg/tools/collect_dataset.sh
echo "Affordance Dataset Creation is Completed" 

echo "Affordance Network Training is Starting" 
bash affordance_seg/tools/train.sh
echo "Affordance Network Training is Compeleted" 
