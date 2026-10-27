//! Unit-test helpers.

/// Defines a `#[test]` that runs `$body(&client)` on every compiled-in CubeCL runtime
/// (through [`corridorkey_core::compute::ComputeTask`]), so tests never name a runtime.
macro_rules! runtime_test {
    ($name:ident, $body:ident) => {
        #[test]
        fn $name() {
            struct Task;
            impl corridorkey_core::compute::ComputeTask for Task {
                type Output = ();
                fn run(self, client: cubecl::prelude::Client) {
                    $body(&client)
                }
            }
            let targets = corridorkey_core::compute::ComputeTarget::available();
            assert!(!targets.is_empty(), "no CubeCL runtime compiled in");
            for target in targets {
                target.run(Task).expect("compiled-in runtime");
            }
        }
    };
}
