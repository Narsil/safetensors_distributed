{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in
        with pkgs;
        {
          default = mkShell {
            nativeBuildInputs = [ pkg-config ];
            buildInputs = [
              rustup
              openssl
              python3Packages.python
              python3Packages.venvShellHook
              cloc
            ] ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [ cudaPackages.cudatoolkit ]);
            venvDir = "./.venv";
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
            LD_LIBRARY_PATH =
              if stdenv.isDarwin then
                "${stdenv.cc.cc.lib}/lib"
              else
                "${stdenv.cc.cc.lib}/lib:${cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib";
          };

        }
      );
    };
}
