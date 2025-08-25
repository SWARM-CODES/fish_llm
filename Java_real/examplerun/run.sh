source /storage/home/hcoda1/9/xzhou473/miniconda3/etc/profile.d/conda.sh
conda activate java21

rm -f ./DB_tempdir/particle_history.db
java -jar ichthyop-3.3.17-LLM.jar example.xml

