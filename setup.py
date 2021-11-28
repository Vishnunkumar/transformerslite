from distutils.core import setup

setup(
  name = 'transformerslite',         
  packages = ['transformerslite'],   
  version = '0.3',      
  license='MIT',        
  description = 'Process, Train and Deploy small transfomer models with less code',   
  author = 'Vishnu N',                  
  author_email = 'vishnunkumar25@gmail.com',      
  url = 'https://github.com/Vishnunkumar/transformerslite',   
  download_url ='https://github.com/Vishnunkumar/transformerslite/archive/refs/tags/v-0.3.tar.gz',    
  keywords = ['NLP', 'Deep learning', 'Transformers'],   
  install_requires = [            
          'transformers'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)