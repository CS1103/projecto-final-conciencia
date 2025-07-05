{ pkgs ? import <nixpkgs> {} }:

let
  llvm = pkgs.llvmPackages_latest;
in
pkgs.clangStdenv.mkDerivation {
  name = "popn_test";

  nativeBuildInputs = with pkgs; [
    cmake
    llvm.lldb
    clang-tools
    llvm.clang
    llvmPackages.openmp
  ];
  buildInputs = with pkgs; [
    llvm.libcxx
    opencv
    ffmpeg-full
    pipewire
    pkg-config
    glib
    libsysprof-capture
    libuuid
  ];
}
