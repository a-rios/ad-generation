#!/bin/bash

### Get temporary API key. legacy key doesn't seem to work properly on the VM, maybe because of the reverse proxy (?)
LABEL_STUDIO_URL='label-studio-url'
# personal access token
PESRONAL_ACCESS_TOKEN="label-studio-personal-access-token"

local_backup_dir=path-to-local-backup-dir


refresh_token=$(curl -X POST $LABEL_STUDIO_URL/api/token/refresh \
    -H "Content-Type: application/json" \
    -d "{\"refresh\": \"${PESRONAL_ACCESS_TOKEN}\"}"| python3 -c 'import sys, json; print(json.load(sys.stdin).get("access",""))'
)

[ -n "$refresh_token" ] && [ "$refresh_token" != "null" ] || { echo "no temporaray access token"; exit 1; }

echo "temporary token: $refresh_token"


for project_id in ... # (fill in project ids from label studio (integer))
do

    date=$(date -I)
    echo "date $date"
    safe_file=$local_backup_dir/$project_id/$date.P$project_id.json
    mkdir -p $local_backup_dir/$project_id
    echo "safe file $safe_file"

    # create export snapshot
    export_id=$(curl -X POST "$LABEL_STUDIO_URL/api/projects/$project_id/exports/" \
     -H "Authorization: Bearer  $refresh_token" \
     -H "Content-Type: application/json" \
     -d '{"exportType":"JSON"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])'
     )
     echo "export id: $export_id"

     while :; do
        status=$(
            curl -sS "$LABEL_STUDIO_URL/api/projects/$project_id/exports/$export_id" \
            -H "Authorization: Bearer $refresh_token" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])'
        )
        [ "$status" = "completed" ] && break
        [ "$status" = "failed" ] && { echo "export failed"; exit 1; }
        sleep 2
        done

        echo "status: $status"

        # download snapshot
        curl "$LABEL_STUDIO_URL/api/projects/$project_id/exports/$export_id/download" \
        -H "Authorization: Bearer $refresh_token" | python3 -m json.tool --indent 2 > $safe_file
        #-o $safe_file

        echo "backing up on remote storage..."
        rsync -azP -e "ssh -i path-to-ssh-key" $safe_file username@remote-host:path-to-remote-backup-dir

done


