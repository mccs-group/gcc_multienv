filegroup(
	name = "gcc-multienv-service-files",
	srcs = [
		"gcc_multienv_service.py",
	],
    visibility = ["//visibility:public"],
)

genrule(
    name = "gcc-multienv-service-bin",
    srcs = [
        ":gcc-multienv-service-files",
    ],
    outs = [
        "gcc-multienv-service",
    ],
    cmd = "cp $(location :gcc-multienv-service-files) $@ && " +
        "chmod 666 $@",
    visibility = ["//visibility:public"],
)
