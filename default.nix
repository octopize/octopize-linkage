let
  pkgs =
    import
      (fetchTarball "https://github.com/b-rodrigues/nixpkgs/archive/d4f50d7ecf7b25cc6f52638cc53231e6bd0dcf64.tar.gz")
      { };

  # Common python dependencies to use in my intermediary inputs
  python_pkgs = pkgs.python312.withPackages (ps: with ps; [
    pandas
    saiph
    seaborn
    matplotlib
    pytest
    numpy
    gower
    octopize_avatar
    ]);
in
pkgs.mkShell {
  LOCALE_ARCHIVE = if pkgs.system == "x86_64-linux" then "${pkgs.glibcLocales}/lib/locale/locale-archive" else "";
  LANG = "en_US.UTF-8";
   LC_ALL = "en_US.UTF-8";
   LC_TIME = "en_US.UTF-8";
   LC_MONETARY = "en_US.UTF-8";
   LC_PAPER = "en_US.UTF-8";
   LC_MEASUREMENT = "en_US.UTF-8";

  buildInputs = [ python_pkgs ];
  
}