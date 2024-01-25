rm nohup.out

python_path="/home/ltan/anaconda3/envs/conceptlime/bin/python"
code_file="/home/ltan/LTan/ConceptLIME/LIME_fidel.py"
path_to_pkl="/home/ltan/LTan/ConceptLIME/Result/LIME_fidel.pickle"
current_datetime="$(date +%Y%m%d%H%M%S)"
direc="Result/$current_datetime"
mkdir $direc

nohup $python_path $code_file

cp nohup.out $direc
cp $code_file $direc

# path_to_pkl=$(find Result/ -name "*.pickle" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d " " -f 2-)

cp $path_to_pkl $direc

# ==================================================
