#bash file to push to git while code refactoring
git pull
git add .
git commit -m "$1"
git push 

