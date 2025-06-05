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
          default =
            mkShell {
              nativeBuildInputs = [ pkg-config ];
              buildInputs = [
                rustup
                openssl
                python3Packages.python
                python3Packages.venvShellHook
              ] ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [ cudaPackages.cudatoolkit ]);
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
            }
            // (pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib:${cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib";
            })
            // (pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
              LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
            });

        }
      );
    };
}
