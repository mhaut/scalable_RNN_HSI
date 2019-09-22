
#####################################
#### CLASSIFICATION EXPERIMENTS #####
#####################################
for dset in indian pavia salinas
do
	for idtest in 0 1 2 3 4
	do
		python -u rnn.py --dataset $dset --idtest $idtest
		python -u rnn.py --dataset $dset --idtest $idtest --vanillarnn
		python -u rnn.py --dataset $dset --idtest $idtest --vanillarnn --cudnn
		python -u rnn.py --dataset $dset --idtest $idtest --lstm
		python -u rnn.py --dataset $dset --idtest $idtest --lstm --cudnn
		python -u rnn.py --dataset $dset --idtest $idtest --gru
		python -u rnn.py --dataset $dset --idtest $idtest --gru --cudnn
	done
done


#####################################
###### SCALABILITY EXPERIMENTS ######
#####################################
for idtest in 0 1 2 3 4
do
	for dset in indian salinas
	do
		for pcabands in 50 100 150
		do
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --vanillarnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --vanillarnn --cudnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --lstm
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --lstm --cudnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --gru
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --gru --cudnn
		done
	done
	for dset in pavia
	do
		for pcabands in 10 40 80
		do
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --vanillarnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --vanillarnn --cudnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --lstm
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --lstm --cudnn
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --gru
			python -u rnn.py --dataset $dset --idtest $idtest --pca $pcabands --gru --cudnn
		done
	done
done


#####################################
######## CLASSIFICATION MAPS ########
#####################################
for dset in indian pavia salinas
do
	python -u rnn.py --dataset $dset --idtest 0 --export_img
	python -u rnn.py --dataset $dset --idtest 0 --vanillarnn --export_img
	python -u rnn.py --dataset $dset --idtest 0 --vanillarnn --cudnn --export_img
	python -u rnn.py --dataset $dset --idtest 0 --lstm --export_img
	python -u rnn.py --dataset $dset --idtest 0 --lstm --cudnn --export_img
	python -u rnn.py --dataset $dset --idtest 0 --gru --export_img
	python -u rnn.py --dataset $dset --idtest 0 --gru --cudnn --export_img
done
