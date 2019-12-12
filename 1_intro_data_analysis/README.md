# Udacity_intro_data_analysis
udacity intro class about data analysis
source 


 check for new courses
    -   Statistics
    - A/B Test
    - machine learning
    
Command libs:
    - conda cmd:
        -Conda installed packages ( including python version):
            conda lsit

        - Conda update packages
            conda update --all  

        - Conda package upgrade:
            conda upgrade conda
            conda upgrade --all

        - Conda install packages:
            conda install numpy scipy pandas

        - Search for package when one does not know the env_nam
            conda search '*beautifulsoup*'

        -create virtual environments:
            - create: 
                conda create -n env_name list of package : conda create -n my_env numpy or conda create -n py2 python=2;
            - activate/deactivate :
                conda activate py2 / deactivate
            - list:
                conda env list
            - export:
                conda env export > environment.yaml
            - import:
                conda env create -f environment.yaml

            - remove:
                conda env remove -n py3 --all
        
        -Jupyther notebook
            - install
                conda install jupyter notebook
                conda install nb_conda
            - open
                jupyter notebook
            - markdown (text)
                https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
            - magic commands
                https://ipython.readthedocs.io/en/stable/interactive/magics.html
        
        about requierments
            - build requierements.txt:
                pip install pipreqs
                pipreqs /path/to/project
            - install requierments:
                pip install -r /path/to/requirements.txt

pandas lib:
    - df.loc, df.iloc documentation: https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
