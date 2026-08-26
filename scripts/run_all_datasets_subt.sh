model_name="subt"
lr=0.0125
num_shots=16

gpus=(0)

scripts=(
  
  "urban_sound.sh"
  # "esc50_actions.sh"
  # "esc50.sh"
  # "ravdess.sh"
  # "tut.sh"
  # "crema_d.sh"
  # "gt_music_genre.sh"
  # "vocal_sound.sh"
  # "sesa.sh"
  # "ns_instruments.sh"
  # "beijing_opera.sh"
)
for i in "${!scripts[@]}"; do
  gpu=${gpus[i]}
  script=${scripts[i]}
  echo "Running ${script} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/"$script" --model_name "$model_name" --lr "$lr" --num_shots "$num_shots" --is_half_labels &
  sleep 5
done

