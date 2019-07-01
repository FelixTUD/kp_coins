for i in {1..200}
do
	weights_file=$(printf "%04d.model" $i)
	base_path="../playground_felix/trained_net/Jun06_15-47-47-889438_taurusa7non_gen_b96_db2_hs100_fc_hd100_lstm_s16_e200_c[1, 2, 5, 20, 50, 100, 200]_seed1/"
	input_path="$base_path$weights_file"

	plot_title="$(printf "Epoch #%d" $i)"
	output_name=$(printf "tsne_%04d.png" $i)
	output_base_path="images_movie_tsne/Jun06_15-47-47-889438/"
	output_path="$output_base_path$output_name"

	python3 torch_coin.py -hs 100 -s 16 -c 2 -b 10 --top_db 2 -p coin_data/data.hdf5 -fc_hd 80 --weights "$input_path" --seed 1 -m tsne --no_state_dict --save_plot "$output_path" --plot_title "$plot_title"
done
