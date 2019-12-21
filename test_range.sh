MODEL=$1
EPOCH_TO=$2
MODE=$3
#Build the csv file
cat ./csv_files/${MODEL}.csv
python3 prepare_csv.py csv_files/${MODEL}.csv
#Do inference of all the models
for i in $(seq 1 $EPOCH_TO); do
python3 train.py test ./datasets/${MODEL}_${i}_cyclegan faster_rcnn_R_50_C4_3x.yaml 137849393/model_final_f97cb7.pkl ${MODE}_bottle ${MODE} ${i}; 
done
#Generate the graphs
python3 generate_graph.py ${MODEL}
