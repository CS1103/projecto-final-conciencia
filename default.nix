{pkgs ? import <nixpkgs> {}}:

let
  llvm = pkgs.llvmPackages_latest;
in
pkgs.clangStdenv.mkDerivation {
  pname = "popn_ai";
  version = "0.1.0";

  src = ./.;

  nativeBuildInputs = with pkgs; [cmake llvm.clang llvm.lldb llvmPackages.openmp clang-tools];
  buildInputs = with pkgs; [llvm.libcxx opencv ffmpeg-full pipewire pkg-config glib dbus xorg.libX11 xorg.libXtst xorg.libXi];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
  ];

  buildPhase = ''
    cmake . -B build $cmakeFlags
    cmake --build build
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp build/pong_ai $out/bin/
  '';

  meta = with pkgs.lib; {
    description = "Pop'n Music AI with ML from my programming project";
    homepage = "https://github.com/Erizur/pong_ai";
    license = licenses.gpl3;
  };
}
