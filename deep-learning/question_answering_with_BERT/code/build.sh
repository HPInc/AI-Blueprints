red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# echo "${red}red text ${green}green text${reset}"

echo "${green}Running build script...${reset}"
echo ""

echo "${green}Creating demo folder if not exists...${reset}"
mkdir -p ./../demo > /dev/null 2>&1
echo ""
echo "${green}Building app...${reset}"
vite build > /dev/null 2>&1
echo ""
echo "${green}Copying build files to demo folder...${reset}"
cp -r ./dist/* ./../demo > /dev/null 2>&1