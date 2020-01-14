gradient jobs create `
--name hello_world `
--optionsFile config.yaml `
--command ( `
    'sh ci/gradient/setup.sh && ' + `
    'echo hello_world && ' + `
    'sh ci/gradient/cleanup.sh' `
)
