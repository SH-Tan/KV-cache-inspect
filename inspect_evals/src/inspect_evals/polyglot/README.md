Run the following package to install all dependencies:

```bash
# cpp
apt update && apt install -y cmake

# golang
apt-get update && apt-get install -y golang

# java
apt-get update
apt-get install -y openjdk-21-jdk

# javascript
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"

nvm install 20      # download + compile if needed
nvm use 20          # set this shell to Node 20

node -v             # v20.x.x :white_check_mark:
npm  -v             # 10.x.x :white_check_mark:

# rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
     | sh -s -- --profile minimal --default-toolchain stable -y

source "$HOME/.cargo/env"

cargo --version   # cargo 1.77.x (or similar)
rustc --version   # rustc 1.77.x
```